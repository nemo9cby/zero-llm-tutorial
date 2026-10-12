/* Module Web Worker of the workbench: the CodeBot forward passes run here, so the page stays responsive while a
   checkpoint loads or a long generation streams. The protocol is documented in playground-engine.js. */
import { createEngine } from './playground-engine.js';

const engine = createEngine({ post: (msg, transfer) => self.postMessage(msg, transfer || []) });
self.onmessage = (e) => engine.handle(e.data);
self.postMessage({ id: null, kind: 'hello' });
