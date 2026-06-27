from core.llm import llm

def review(question, discussion):

    prompt = f"""
You are the Lead DFT Architect compiling the final production-ready DFT architecture plan.

Question:
{question}

Discussion:
{discussion}

Create the final compiled answer. It must read like an architecture review or senior-level interview defense rather than generic textbook summaries.

Requirements:
- **Production-ready tone**: Use sharp, concise, engineering-oriented bullet points (suitable for a senior tape-out review checklist or senior interview).
- **Enforce exact parameters**: Do NOT summarize details out. Preserve specific numbers (e.g., target scan depths of 500–2000 flops, 30–150× compression), standard names, and specific hardware.
- **Scan Eligibility**: Ensure scan includes only "scan-eligible" sequential elements, explicitly excluding sensitive logic (analog boundaries, pre-PLL, clock-gen).
- **Stitching & Containment**: Enforce that scan chains are partitioned strictly within each clock/power domain, using lockup latches and isolation cells only where crossings are absolutely unavoidable. Do NOT claim scan chains are stitched across domains freely.
- **Clarify EDT/Compression**: Strictly define EDT as mapping many internal chains to fewer external channels to compress data/time (no physical flop reduction). Acknowledge that 30–150× is typical but overly aggressive compaction hurts X-tolerance and diagnosis.
- **At-speed OCC**: Enforce launch-on-capture (broadside) or launch-on-shift at functional speed, using synchronous OCCs for inter-domain paths.
- **Power Gating**: Enforce domain-contained scan paths, level shifters, isolation cells, and test-mode controls (PTAM-like overrides).
- **Memory BIST**: Enforce clustering of small memories, dedicated controllers for large memories, BISR/BIRA, and power-aware scheduling.
- **Static ATE Scheduling**: Emphasize static power-aware test scheduling with limited conditional branching (like early abort on failures) rather than unrealistic "real-time dynamic scheduling".
- **Verification/Signoff**: Include specific sign-off metrics: Scan DRC clean, compression-aware ATPG coverage targets met, and GLS with SDF for scan shift and at-speed OCC modes.
- **Remove duplicates**: Merge overlapping arguments from experts, but preserve the technical specifics.
- **Source Citations**: You MUST collect and list all referenced source documents and page numbers (citations) from the discussion at the end of your compiled plan (under a "Sources & References" section), so the user knows exactly which textbooks or lecture files the information came from.

Return only the final structured architectural answer.
"""

    response = llm.invoke(prompt)

    return response.content