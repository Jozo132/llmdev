/**
 * ByteBpeTokenizer — compact, dependency-free byte-level BPE.
 *
 * - Trains greedy merges on a bounded in-memory sample (never touches disk).
 * - Emits uint16 token ids (vocab ≤ 65535) so token shards pack 2 bytes/token.
 * - Deterministic & serializable ⇒ exportable alongside model weights.
 */
import type { TokenizerHandle } from "../core/types.js";

const enc = new TextEncoder();
const dec = new TextDecoder();

export class ByteBpeTokenizer implements TokenizerHandle {
  /** merges[i] = [left, right] producing token id 256 + i */
  private merges: Array<[number, number]> = [];
  private mergeRank = new Map<string, number>();

  get vocabSize(): number {
    return 256 + this.merges.length;
  }

  /** Train BPE merges on a sample capped by the caller (keep it a few MB). */
  train(sample: string, targetVocab: number): void {
    if (targetVocab > 65535) throw new Error("uint16 vocab cap is 65535");
    let ids: number[] = Array.from(enc.encode(sample));
    this.merges = [];
    this.mergeRank.clear();

    while (256 + this.merges.length < targetVocab) {
      // Count adjacent pairs.
      const counts = new Map<string, number>();
      for (let i = 0; i < ids.length - 1; i++) {
        const k = `${ids[i]},${ids[i + 1]}`;
        counts.set(k, (counts.get(k) ?? 0) + 1);
      }
      let bestKey = "";
      let bestCount = 1; // require count ≥ 2 to be worth a merge
      for (const [k, c] of counts) {
        if (c > bestCount) {
          bestCount = c;
          bestKey = k;
        }
      }
      if (!bestKey) break;

      const [a, b] = bestKey.split(",").map(Number);
      const newId = 256 + this.merges.length;
      this.mergeRank.set(bestKey, this.merges.length);
      this.merges.push([a, b]);

      // Apply the merge in place.
      const out: number[] = [];
      for (let i = 0; i < ids.length; i++) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
          out.push(newId);
          i++;
        } else {
          out.push(ids[i]);
        }
      }
      ids = out;
    }
  }

  encode(text: string): Uint16Array {
    let ids: number[] = Array.from(enc.encode(text));
    // Repeatedly apply the lowest-rank applicable merge (standard BPE encode).
    while (ids.length >= 2) {
      let bestRank = Infinity;
      let bestPos = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.mergeRank.get(`${ids[i]},${ids[i + 1]}`);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestPos = i;
        }
      }
      if (bestPos < 0) break;
      ids.splice(bestPos, 2, 256 + bestRank);
    }
    return Uint16Array.from(ids);
  }

  decode(tokens: ArrayLike<number>): string {
    const bytes: number[] = [];
    const expand = (id: number): void => {
      if (id < 256) {
        bytes.push(id);
      } else {
        const [a, b] = this.merges[id - 256];
        expand(a);
        expand(b);
      }
    };
    for (let i = 0; i < tokens.length; i++) expand(tokens[i]);
    return dec.decode(Uint8Array.from(bytes));
  }

  toJSON(): unknown {
    return { kind: "byte-bpe", merges: this.merges };
  }

  static fromJSON(json: { merges: Array<[number, number]> }): ByteBpeTokenizer {
    const t = new ByteBpeTokenizer();
    t.merges = json.merges;
    json.merges.forEach(([a, b], i) => t.mergeRank.set(`${a},${b}`, i));
    return t;
  }
}
