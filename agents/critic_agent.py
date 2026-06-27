from core.llm import llm

def critique(question, discussion):

    prompt = f"""
Question:
{question}

Discussion:
{discussion}

You are a Principal DFT Architect conducting a strict design review for a production-grade SoC tape-out.

Your task is to analyze the discussion and identify any weaknesses, hand-waving, or technical inaccuracies. Be extremely critical.

Review for the following strict production criteria:
1. **Clock / CDC Timing**: Demands scan chains be partitioned strictly within domains. Reject statements about stitching scan chains across domains; they must be contained within each clock/power domain, using lockup latches only where crossing is unavoidable. Specify target chain depths (500–2000 flops).
2. **Scan Eligibility**: Demands specifying "scan-eligible" sequential elements, explicitly excluding sensitive logic (analog boundaries, pre-PLL logic, clock-generation state machines).
3. **Power Gating / Low-Power DFT**: Demands scan paths local to power domains, using level shifters/isolation cells at boundaries. Reject answers that omit retention/isolation test sequences or test-mode overrides (like PTAM).
4. **EDT / Scan Compression**: Demands defining EDT as mapping internal chains to fewer external channels to compress data/time (no physical flop reduction). Target 30-150x compression, noting that overly aggressive compaction hurts X-tolerance and diagnostics.
5. **At-speed Clocks (OCC)**: Demands specifying launch-on-capture (broadside) or launch-on-shift at functional speed, using synchronous OCCs for inter-domain paths.
6. **Memory Test (MBIST)**: Demands clustering small memories, dedicated controllers for large memories, BIRA/BISR repair integration, and power-aware scheduling.
7. **JTAG / Wrappers**: Demands IEEE 1149.1 for top-level access, and IEEE 1500/1687 (IJTAG) wrappers for large IP blocks.
8. **Test Scheduling**: Reject "real-time dynamic scheduling" on ATE (which is unrealistic). Demand static power-aware schedules, permitting only basic conditional branching (like early abort on failure).
9. **Verification/Signoff**: Demand specific sign-off metrics: Scan DRC clean, compression-aware ATPG coverage targets met, and gate-level simulation (GLS) with SDF for scan shift and at-speed OCC modes.

Tasks:
1. Identify incorrect statements (especially regarding EDT, clocking, or power gating).
2. Identify hand-waving or generic textbook concepts that lack specific engineering numbers or design rules.
3. Identify missing elements based on the criteria above.
4. Suggest concrete improvements.

Return:
Incorrect Points:
Hand-waving & Theory-ish Points:
Missing Concrete Engineering Elements:
Improvement Suggestions:
"""

    response = llm.invoke(prompt)

    return response.content