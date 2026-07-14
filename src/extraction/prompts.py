"""Prompt templates for evidence-grounded problem extraction."""


PROBLEM_EXTRACTION_PROMPT = """
Extract only a genuine user problem that is explicitly supported by the source text.
Return JSON matching the requested schema. Do not infer or invent revenue, market size,
customer type, workflow, willingness to pay, frequency, or severity. Use null fields and
low confidence when the source is insufficient. The evidence quote must be an exact,
short excerpt from the source. Valid pain types are: time, labor, cost, lost_revenue,
risk, compliance, coordination, data_entry, poor_user_experience, lack_of_visibility,
integration, repetitive_work.
""".strip()
