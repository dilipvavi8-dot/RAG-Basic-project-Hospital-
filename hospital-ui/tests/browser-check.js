async (page) => {
  // Browser-only API fixtures: never send a chat to the real model in this check.
  let mode = 'success';
  const sent = [];
  await page.route('**/api/chat', async route => {
    sent.push(route.request().postDataJSON());
    if (mode === 'offline') return route.abort('failed');
    await new Promise(resolve => setTimeout(resolve, 150));
    await route.fulfill({json: {
      answer: 'Browser test fixture [1]\n<img src=x onerror=alert(1)>\nportal.hospital.example',
      sources: [
        {label: 1, filename: 'fixture.txt', section: 'Test', subsection: 'Cited', cited: true},
        {label: 2, filename: 'fixture.txt', section: 'Other', subsection: '', cited: false},
      ], search_used: true, elapsed_seconds: 0.15,
    }});
  });
  await page.goto('http://127.0.0.1:8000');
  await page.getByText('Connected to the hospital RAG pipeline.').waitFor();
  await page.setViewportSize({width: 1440, height: 1000});
  await page.screenshot({path: 'output/playwright/desktop.png', fullPage: true});
  await page.getByRole('button', {name: /FIND A DEPARTMENT/}).click();
  const input = page.getByRole('textbox', {name: 'Your question'});
  if (await input.inputValue() !== 'Where is the cardiology department?') throw Error('Example did not fill input');
  await input.press('Shift+Enter');
  if (!(await input.inputValue()).includes('\n')) throw Error('Shift+Enter failed');
  await input.press('Enter');
  await page.getByText('References · 1 cited / 2 retrieved').waitFor();
  await page.getByText('References · 1 cited / 2 retrieved').click();
  await page.getByText('[2] fixture.txt · Retrieved only').waitFor();
  if (await page.locator('#messages img').count()) throw Error('Unsafe HTML rendered');
  if (await page.locator('#messages a').count()) throw Error('Fictional address became a link');
  const firstThread = sent[0].conversation_id;
  mode = 'offline';
  await input.fill('Retry test');
  await input.press('Enter');
  await page.getByText('Cannot reach the local backend.', {exact: false}).waitFor();
  const count = await page.locator('.message.user').count();
  const requestId = sent[sent.length - 1].request_id;
  mode = 'success';
  await page.getByRole('button', {name: 'Retry', exact: true}).click();
  await page.waitForFunction(() => document.querySelectorAll('.message.assistant').length === 2);
  if (await page.locator('.message.user').count() !== count) throw Error('Retry duplicated user message');
  if (sent[sent.length - 1].request_id !== requestId) throw Error('Retry changed request ID');
  await page.getByRole('button', {name: '＋ New Conversation', exact: true}).click();
  await page.getByText('New conversation ready.', {exact: false}).waitFor();
  if (await page.locator('.message').count()) throw Error('Reset did not clear messages');
  await input.fill('New thread test');
  await input.press('Enter');
  await page.getByText('References · 1 cited / 2 retrieved').waitFor();
  if (sent[sent.length - 1].conversation_id === firstThread) throw Error('Reset reused old ID');
  await page.setViewportSize({width: 390, height: 844});
  await page.screenshot({path: 'output/playwright/mobile.png', fullPage: true});
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw Error('Mobile horizontal overflow');
  await page.goto('http://127.0.0.1:8000/guide/');
  await page.locator('img').waitFor();
  if (!await page.locator('img').evaluate(img => img.complete && img.naturalWidth > 0)) throw Error('Diagram image failed');
  await page.setViewportSize({width: 1200, height: 1100});
  await page.screenshot({path: 'output/playwright/guide.png', fullPage: true});
  console.log('PASS: examples, Enter/Shift+Enter, references, safe output, disconnected retry, new ID, mobile width, diagram image. Browser chat responses were fixtures; no model calls.');
}
