import test from "node:test";
import assert from "node:assert/strict";
import {LatestJob} from "../web/scheduler.mjs";

function fixture() {
  const sent = [], received = [];
  return {sent, received, jobs: new LatestJob(job => sent.push(job), message => received.push(message))};
}
test("coalesces scrubs to one latest pending job and rejects stale results", () => {
  const {jobs, sent, received} = fixture();
  jobs.submit("analyze", {start: 0});
  for (let start = 1; start <= 1000; start++) jobs.submit("analyze", {start});
  assert.equal(sent.length, 1);
  jobs.complete(sent[0]);
  assert.equal(received.length, 0);
  assert.equal(sent.length, 2);
  assert.equal(sent[1].payload.start, 1000);
  jobs.complete(sent[1]);
  assert.equal(received.length, 1);
  assert.equal(jobs.active, null);
});
test("changing recording rejects old open and analyze responses", () => {
  const {jobs, sent, received} = fixture();
  jobs.submit("open", {name: "old"});
  jobs.invalidate();
  jobs.submit("open", {name: "new"});
  jobs.complete(sent[0]);
  assert.equal(received.length, 0);
  jobs.complete(sent[1]);
  assert.equal(received[0].payload.name, "new");
  jobs.submit("analyze", {});
  jobs.invalidate();
  jobs.complete(sent[2]);
  assert.equal(received.length, 1);
});
test("duplicate or unmatched replies cannot release an active job", () => {
  const {jobs, sent} = fixture();
  jobs.submit("open", {});
  jobs.complete({id: -1});
  assert.equal(jobs.active.id, sent[0].id);
  jobs.complete(sent[0]);
  jobs.submit("analyze", {});
  jobs.complete(sent[0]);
  assert.equal(jobs.active.id, sent[1].id);
});
test("invalidation clears pending operations while a fetch is in progress", () => {
  const {jobs, sent, received} = fixture();
  jobs.submit("analyze", {});
  jobs.submit("analyze", {start: 100});
  jobs.invalidate();
  jobs.complete({...sent[0], error: "old failure"});
  assert.equal(sent.length, 1);
  assert.equal(received.length, 0);
  jobs.submit("open", {});
  assert.equal(sent.length, 2);
});
