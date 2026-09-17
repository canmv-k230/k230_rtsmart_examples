import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const header = readFileSync(new URL('../web_page.h', import.meta.url), 'utf8');
const html = [...header.matchAll(/^\s*("(?:[^"\\]|\\.)*")/gm)]
  .map(match => JSON.parse(match[1])).join('');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

async function waitFor(predicate) {
  for (let i = 0; i < 20 && !predicate(); i++) {
    await new Promise(resolve => setImmediate(resolve));
  }
  assert.ok(predicate());
}

function setup({ holdRelease = false, holdAnswer = false } = {}) {
  const elements = {}, peers = [], offers = [], answers = [], releases = [];
  class Peer {
    constructor() {
      this.id = peers.length + 1;
      this.iceGatheringState = 'complete';
      this.closed = false;
      peers.push(this);
    }
    close() { this.closed = true; }
    async setRemoteDescription() { if (this.remoteGate) await this.remoteGate.promise; }
    async createAnswer() { return { sdp: `answer-peer-${this.id}` }; }
    async setLocalDescription(answer) { this.localDescription = answer; }
  }
  const context = vm.createContext({
    URLSearchParams, encodeURIComponent, setTimeout, clearTimeout,
    document: {
      getElementById(id) {
        return elements[id] ||= { style: {}, classList: { add() {} }, disabled: false };
      },
    },
    window: { location: { search: '' }, addEventListener() {} },
    navigator: { sendBeacon() {} },
    RTCPeerConnection: Peer,
    fetch(url, options) {
      const pending = deferred();
      const request = { url, options, ...pending };
      if (url === '/offer') offers.push(request);
      else if (url.startsWith('/answer?')) {
        answers.push(request);
        if (!holdAnswer) pending.resolve({ ok: true });
      } else if (url.startsWith('/disconnect?')) {
        releases.push(request);
        if (!holdRelease) pending.resolve({ ok: true });
      } else throw new Error(`Unexpected request: ${url}`);
      return pending.promise;
    },
  });
  vm.runInContext(script, context);
  function replyOffer(index, session) {
    offers[index].resolve({
      ok: true, headers: { get: () => session }, text: async () => 'offer',
    });
  }
  const currentPeer = () => vm.runInContext('pc', context);
  return { context, currentPeer, elements, peers, offers, answers, releases, replyOffer };
}

async function testRepeatedClicksDuringRelease() {
  const app = setup({ holdRelease: true });
  vm.runInContext("pc=new RTCPeerConnection();pc.sessionId='old';doDisconnect();", app.context);
  const first = app.context.doConnect();
  const second = app.context.doConnect();
  assert.equal(app.elements.btn.disabled, true);
  assert.equal(app.offers.length, 0);
  app.releases[0].resolve({ ok: true });
  await waitFor(() => app.offers.length === 1);
  app.replyOffer(0, 'new');
  await Promise.all([first, second]);
  assert.equal(app.peers.length, 2);
  assert.equal(app.answers.length, 1);
  assert.ok(app.answers[0].url.includes('session=new'));
  assert.equal(app.answers[0].options.body, 'answer-peer-2');
}

async function testLateOfferCannotReplaceCurrentSession() {
  const app = setup();
  const first = app.context.doConnect();
  await waitFor(() => app.offers.length === 1);
  app.context.doDisconnect();
  const second = app.context.doConnect();
  await waitFor(() => app.offers.length === 2);
  app.replyOffer(1, 'new');
  await second;
  app.replyOffer(0, 'old');
  await first;
  assert.equal(app.currentPeer().sessionId, 'new');
  assert.equal(app.currentPeer().closed, false);
  assert.equal(app.answers.length, 1);
  assert.ok(app.answers[0].url.includes('session=new'));
  assert.equal(app.releases.length, 1);
  assert.ok(app.releases[0].url.includes('session=old'));
}

async function testLateFailureCannotCloseNewPeer() {
  const app = setup({ holdAnswer: true });
  const first = app.context.doConnect();
  await waitFor(() => app.offers.length === 1);
  app.replyOffer(0, 'old');
  await waitFor(() => app.answers.length === 1);
  app.context.doDisconnect();
  const second = app.context.doConnect();
  await waitFor(() => app.offers.length === 2);
  app.replyOffer(1, 'new');
  await waitFor(() => app.answers.length === 2);
  app.answers[1].resolve({ ok: true });
  await second;
  app.answers[0].reject(new Error('old request failed'));
  await first;
  assert.equal(app.currentPeer().sessionId, 'new');
  assert.equal(app.currentPeer().closed, false);
  assert.notEqual(app.elements['status-text'].textContent, 'failed');
  assert.equal(app.releases.length, 1);
}

async function testDisconnectDuringNegotiationSuppressesAnswer() {
  const app = setup();
  const connecting = app.context.doConnect();
  await waitFor(() => app.offers.length === 1);
  const gate = app.peers[0].remoteGate = deferred();
  app.replyOffer(0, 'old');
  await waitFor(() => app.currentPeer().sessionId === 'old');
  app.context.doDisconnect();
  gate.resolve();
  await connecting;
  assert.equal(app.answers.length, 0);
  assert.equal(app.releases.length, 1);
  assert.equal(app.currentPeer(), null);
  assert.equal(app.elements.btn.disabled, false);
}

for (const test of [testRepeatedClicksDuringRelease,
  testLateOfferCannotReplaceCurrentSession, testLateFailureCannotCloseNewPeer,
  testDisconnectDuringNegotiationSuppressesAnswer]) {
  await test();
  console.log(`${test.name}: passed`);
}
