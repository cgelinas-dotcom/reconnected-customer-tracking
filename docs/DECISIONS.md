# Decisions to make

Three forks in your 11-step plan that need an answer from you before the relevant phase. You don't need to answer now — they're listed here so we remember to come back to them.

---

## Fork 1 — Multi-store networking (Phase 7)

**The question:** How does this Mac (or eventually a server) reach the NVRs at all 8 stores?

**Already decided:** Centralized processing (one box pulls all streams), not edge processing (per-store boxes).

**Options for the connection itself:**

- **Tailscale** *(recommended)* — install on a tiny computer at each store (Mac mini, Raspberry Pi, or even the store's existing PC). The central box joins the same Tailnet and reaches every NVR by its Tailscale IP. Free for personal use up to 100 devices. ~30 min setup per store.
- **Site-to-site VPN** — more "enterprise," more brittle, more router config. Skip unless Tailscale is blocked.
- **Port forwarding** *(do not do this)* — exposes the NVR to the public internet. Security nightmare.

**Recommendation:** Tailscale. Defer the actual setup until Phase 7. Right now, just pick a store on the same network as this Mac for testing.

---

## Fork 2 — Employee identification (Phase 2)

**The question:** How does the system know a person on camera is staff (don't count them) vs a customer (do count them)?

**Options:**

- **Face recognition** — enroll each staff member's face (a few photos), system auto-identifies them when they're on camera. **Pros:** very accurate, works even if staff aren't in uniform. **Cons:** stores facial biometric data of your employees — needs a signed consent form, and in some states (Illinois BIPA, Texas, Washington) carries legal weight. You'd need a written employee consent + retention policy. Privacy implications are real.
- **Visual markers** — staff wear branded lanyards / vests / hats. System detects the marker and treats anyone wearing one as staff. **Pros:** zero biometric data, easy to explain to employees, easy to opt in/out (just don't wear the lanyard). **Cons:** less accurate (markers occlude, fall off, get forgotten), staff have to actually wear them.
- **Both** — face primary, marker as fallback. Most accurate, most complex, still has the biometric consent issue.
- **Defer** — build the system without an employee filter at first. Count *everyone*. Later, subtract your known schedule (staff X was working 9–5, subtract ~Y events). Useful for getting started — terrible long-term.

**Recommendation for the build:** start with **defer** so we can see raw output, then pick **visual markers** (lanyards) as the long-term answer unless you have a specific reason to want face recognition. The biometric consent paperwork is a real cost you'd rather not take on if a lanyard solves it.

**Your call when we get to Phase 2.**

---

## Fork 3 — "Unique customer" definition (Phase 4)

**The question:** When the system sees the same person twice, when is it "one customer" vs "two visits"?

**Options:**

- **Session with timeout** — same person seen again within N minutes = same visit. Gap longer than N = new visit. Typical N: 30 or 60 minutes. Captures "customer wandered out to their car and came back."
- **Per day** — same person all day = 1 customer that day. Simple, conservative. Matches how most retail counts foot traffic.
- **Per visit, no merging** — every distinct presence is its own count. Inflates numbers.
- **Track all three; decide at query time** *(recommended for the build)* — log every detection with timestamps and an identity. The dashboard offers "unique per session," "unique per day," and "raw visits" as toggles. You don't have to commit; you see all three numbers and pick the one that matches reality.

**Recommendation:** track all three, expose toggles in the dashboard. Once you see real numbers across a real week, the right default becomes obvious.

**Your call when we get to Phase 4.** Defaulting to track-all-three until you say otherwise.

---

## A privacy note we'll need eventually

Whatever you pick for Fork 2, posting a sign at the door is a good idea (and required in some jurisdictions): "Video monitoring in use. Customer counts collected for operational purposes. No video footage shared." Keep it boring and accurate. We'll draft one when we hit Phase 6.
