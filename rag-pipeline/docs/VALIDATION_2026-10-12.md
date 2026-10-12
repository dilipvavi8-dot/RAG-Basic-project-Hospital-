# Live RAG and UI validation — October 12, 2026

## Conclusion

The UI backend successfully calls the real LangGraph agent, retriever, persisted Chroma index, and configured OpenAI model. The core factual lookups work in this sample. The pipeline does not yet satisfy every expected routing and response-quality rule; this is not a claim of perfect accuracy.

This run used the current configured models (`gpt-6-luna` for chat and `text-embedding-3-small` for embeddings), current prompts, and existing hospital index. No model/prompt/retrieval changes were made to improve the measured results. These were paid-capable live calls, not simulated answers.

## Measured results

- 11 configured test cases, containing 15 question turns, all completed through the real `/api/chat` endpoint with HTTP 200.
- Search routing matched the configured expectation on 11 of 14 applicable checks. The symptom question has no required search-routing value.
- Every emitted citation label resolved to a returned source entry. Turns without citations also passed the label-validity check; this does not independently prove evidential support.
- Backend response time: 1.06 seconds minimum, 3.07 seconds median, 7.11 seconds maximum in this run.
- Seven mocked backend tests also passed. These test API mechanics and concurrency, not answer correctness.
- One additional real browser chat successfully returned dermatology Saturday hours (9:00 AM–1:00 PM) with a cited Dermatology Overview reference. The reference panel opened and New Conversation cleared visible messages. This extra browser check is separate from the 15-turn dataset totals above.

## Answer review against the hospital document

| Question/behavior | Observed result | Assessment |
| --- | --- | --- |
| Cardiology location | East Tower, 3rd Floor, citing the Cardiology Overview passage | Correct documented fact |
| Cardiology initial consultation | $250, explicitly a self-pay estimate | Correct documented fact |
| Dermatology Saturday hours | 9:00 AM–1:00 PM, citing Dermatology Overview | Correct documented fact |
| Unknown Dr. Smith | Said the direct number was not found; offered the general appointment line with a citation | No invented doctor number |
| Appointment slot | Said it cannot check live availability; supplied documented contact instructions | Correct limitation |
| Greeting, both cases | Brief reply with no search | Correct routing |
| Ambiguous hours in a fresh conversation | Asked which department, but searched first | Routing failure |
| Greeting followed by ambiguous hours | Asked which department and gave general hours, but searched first | Routing failure |
| Dermatology location followed by “its hours” | Correctly maintained dermatology context and retrieved its hours | Follow-up memory worked |
| “Book an appointment there” follow-up | Used dermatology context, cited the portal/appointment line and referral policy; did not claim booking success | Correct documented instructions |
| Ambiguous hours after reset | Asked which department; did not assume prior dermatology context, but searched first | Reset isolation worked; routing failed |
| “Leg pain doctor” | Supplied Orthopedics contact details directly | Contact facts are supported; referral wording needs qualification |
| Orthopedics online consultation | Said documents list telehealth “only” for named departments, “not orthopedics” | Potentially overstates an incomplete list; should explicitly say availability is not confirmed |

The symptom and telehealth rows are review concerns, not fabricated numeric accuracy failures. Answer factuality, citation validity, and tool-routing compliance are different measures.

## Problems found and how to fix them

### 1. UI moved inside the RAG folder

The old relative `RAG_PROJECT_PATH` pointed to a nonexistent sibling. The UI `.env` and `.env.example` now use `RAG_PROJECT_PATH=..`. The moved virtual environment's activation script also contained the old absolute path. Re-running `python3 -m venv hospital-ui/.venv` refreshed that environment's activation scripts. The local UI setup instructions were updated for the nested location.

### 2. Ambiguous questions trigger unnecessary searches

The current model decides whether to search using prompt instructions. `route_after_agent` follows any emitted tool call; `search_node` verifies tool name and search count, but does not verify that the query has a resolved topic. Thus a prompt instruction to clarify first is not enforced by graph logic.

Suggested fix: explicitly represent a resolved department/topic in state and validate the decision before allowing a search. When neither the current message nor conversation history identifies a topic, return a clarification request. Add regression checks for the three ambiguous cases and for successful follow-ups, so clarification rules do not accidentally disable valid memory-based queries. Avoid a blanket rule rejecting every pronoun: “its” is appropriate when a department is already established.

### 3. Symptom-to-specialty wording

The source lists departments, not a triage rule for every symptom. Strengthen the prompt to ask what department the user is looking for or provide a clearly conditional directory option, without implying that the named specialty is medically appropriate. Keep any contact details retrieved and cited. Re-run the existing `leg_pain_department` case.

### 4. Missing information versus negative claims

The document lists several telehealth departments but does not explicitly exclude all others. Require wording such as “The documents do not confirm whether Orthopedics offers video consultations.” Offer a cited contact for confirmation. Add a prompt example distinguishing online booking from video visits and rerun `orthopedics_online`.

These agent changes are recommendations only. The current model, prompt, graph, retrieval, and evaluation behavior were preserved to make this a current-configuration validation.

## Limits

This is one run of a small fictional dataset. Responses may change on later runs. A valid citation number alone does not prove that every sentence is supported. Saved raw responses are in `data/evaluation_results/ui_live_validation_20261012.json`. No real appointment system, booking operation, persistent chat memory, or production deployment was tested.
