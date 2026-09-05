/** One running operation and one replaceable pending operation. */
export class LatestJob {
  constructor(send, receive) {
    this.send = send;
    this.receive = receive;
    this.generation = 0;
    this.serial = 0;
    this.latest = 0;
    this.active = null;
    this.pending = null;
  }
  invalidate() {
    this.generation += 1;
    this.pending = null;
    this.latest = ++this.serial;
    return this.generation;
  }
  submit(type, payload) {
    const job = {id: ++this.serial, generation: this.generation, type, payload};
    this.latest = job.id;
    this.pending = job;
    this.pump();
    return job.id;
  }
  pump() {
    if (this.active || !this.pending) return;
    this.active = this.pending;
    this.pending = null;
    this.send(this.active);
  }
  complete(message) {
    if (!this.active || message.id !== this.active.id) return;
    this.active = null;
    if (message.generation === this.generation && message.id === this.latest) {
      this.receive(message);
    }
    this.pump();
  }
}
