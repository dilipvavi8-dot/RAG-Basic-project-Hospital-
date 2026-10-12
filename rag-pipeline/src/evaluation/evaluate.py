import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from src.agents.agent_graph import build_graph


def conversation_config():
    return {
        "configurable": {
            "thread_id": str(uuid4()),
        }
    }


def save_report(report, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def display_list(title, items, empty_message):
    print(f"\n{title}")

    if items:
        for item in items:
            print("-", item)
    else:
        print("-", empty_message)


def read_review():
    while True:
        verdict = input(
            "\nYour review: pass / fail / unsure: "
        ).strip().lower()

        if verdict in {"pass", "fail", "unsure"}:
            break

        print("Please type pass, fail, or unsure.")

    note = input("Optional review note: ").strip()
    return verdict, note


def print_summary(report, report_path):
    results = report["results"]

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("Run status:", report["status"])
    print("Reviewed questions:", len(results))

    for verdict in ["pass", "fail", "unsure"]:
        count = sum(
            item["verdict"] == verdict for item in results
        )
        print(f"{verdict.title()}: {count}")

    checked = [
        item for item in results
        if item["routing_match"] is not None
    ]
    mismatches = sum(
        not item["routing_match"] for item in checked
    )

    print(
        f"\nSearch behavior mismatches: "
        f"{mismatches} / {len(checked)} checked"
    )

    print("\nIndividual results:")
    for item in results:
        routing_note = ""

        if item["routing_match"] is False:
            routing_note = " | SEARCH BEHAVIOR MISMATCH"

        print(
            f"{item['case_id']} / {item['step_id']}: "
            f"{item['verdict']}{routing_note}"
        )

        if item["note"]:
            print("  Note:", item["note"])

    print("\nReport saved to:", report_path)


def evaluate():
    questions_path = (
        PROJECT_ROOT / "data" / "sample_queries" / "questions.json"
    )
    test_cases = json.loads(
        questions_path.read_text(encoding="utf-8")
    )

    # Supports both the original single-question format and
    # the new conversation format.
    cases = [
        {
            "id": case["id"],
            "steps": case.get("steps", [case]),
        }
        for case in test_cases
    ]

    total_questions = sum(len(case["steps"]) for case in cases)

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid4().hex[:8]
    )
    report_path = (
        PROJECT_ROOT
        / "data"
        / "evaluation_results"
        / f"evaluation_{run_id}.json"
    )

    report = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "test_cases": len(cases),
        "planned_questions": total_questions,
        "results": [],
    }

    graph = build_graph(enable_memory=True)

    print("Hospital RAG Evaluation — Conversations")
    print(f"Test cases: {len(cases)}")
    print(f"Questions: {total_questions}")
    print("Review factual accuracy, citations, and expected behavior.")
    print("Search behavior is checked separately.")
    print("Model and embedding calls may incur API charges.")
    print("Completed reviews are saved as you proceed.\n")

    save_report(report, report_path)

    try:
        for case_number, case in enumerate(cases, start=1):
            # Each test case starts with independent history.
            config = conversation_config()

            print("\n" + "=" * 60)
            print(
                f"Test case {case_number}: {case['id']}"
            )

            for step_number, step in enumerate(
                case["steps"], start=1
            ):
                if step.get("reset_before", False):
                    config = conversation_config()
                    print("\nConversation reset before this question.")

                step_id = step.get(
                    "id", f"step_{step_number}"
                )

                print("\n" + "-" * 60)
                print(f"Step {step_number}: {step_id}")
                print("Question:", step["question"])

                result = graph.invoke(
                    {"question": step["question"]},
                    config=config,
                )

                answer = result["answer"]
                sources = result.get("sources", [])
                actual_search = bool(result.get("searched", False))
                expected_search = step.get("expected_search")

                routing_match = (
                    None
                    if expected_search is None
                    else actual_search == expected_search
                )

                print("\nActual answer:")
                print(answer)

                print("\nExpected behavior:")
                print(step["expected_behavior"])

                display_list(
                    "Expected facts:",
                    step.get("expected_facts", []),
                    "No specific facts required.",
                )
                display_list(
                    "Must not claim:",
                    step.get("must_not_claim", []),
                    "No additional restrictions.",
                )

                print("\nSearch behavior:")
                print("Actual search:", actual_search)

                if expected_search is not None:
                    print("Expected search:", expected_search)
                    print(
                        "Routing check:",
                        "PASS" if routing_match else "FAIL",
                    )
                else:
                    print("Routing check: Not specified.")

                if sources:
                    print("\nRetrieved passages — reference key:")

                    for source in sources:
                        print(
                            f"[{source['label']}] "
                            f"{source['filename']} | "
                            f"{source['section']} | "
                            f"{source['subsection']}"
                        )
                else:
                    print("\nNo retrieved passages.")

                verdict, note = read_review()

                report["results"].append({
                    "case_id": case["id"],
                    "step_id": step_id,
                    "reset_before": step.get("reset_before", False),
                    "question": step["question"],
                    "answer": answer,
                    "sources": sources,
                    "expected_behavior": step["expected_behavior"],
                    "expected_facts": step.get("expected_facts", []),
                    "must_not_claim": step.get("must_not_claim", []),
                    "expected_search": expected_search,
                    "actual_search": actual_search,
                    "routing_match": routing_match,
                    "verdict": verdict,
                    "note": note,
                })

                save_report(report, report_path)

        report["status"] = "completed"

    except (KeyboardInterrupt, EOFError):
        report["status"] = "interrupted"
        print("\nEvaluation stopped. Completed reviews are preserved.")

    except Exception:
        report["status"] = "failed"
        raise

    finally:
        report["finished_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        save_report(report, report_path)
        print_summary(report, report_path)


if __name__ == "__main__":
    evaluate()