"""Choi et al. (2026) templates copied verbatim from util/templates/VoteTemplates.py and Framework/RAG_RFLX_GRAPH.py
(commit 5a52f82). Only {answer_format} is replaced: option-likelihood scoring for final answers, a short JSON
instruction for rehearsal (replaces PydanticOutputParser instructions)."""
import json, re, ast

BASIC = """
You are a representative of {country} in United Nations Security Council(UNSC).
You have to decide to to vote on following <draft_resolution> in UNSC.
You only choose the vote among three options: favour, against or abstention. 
Choose the best option considering your country's interest.


\n\n<draft_resolution> :
{resolution}
 
\n\nFORMAT:
{format}
 
"""

WITH_HISTORY = """
Your answer should be very concise.
You are a representative of {country} in the United Nations Security Council (UNSC). You must decide how to vote on the following <draft_resolution> in the UNSC.
Your decision should be based on  {country}'s diplomatic stance and also its national interest.
You should consider the previous vote prediction information that you gave on similar resolutions in <previous_vote_prediction>
In <previous_vote_prediction>, there are also reflections of your prediction from ground truth, which you can utilize in order to understand more clearly on {country}'s stance and interest on this matter.

**Important guidelines:**  
- Most importantly, do **not** incorporate biases about {country}.
- The vote must be chosen from three options only: **favour, against, or abstention**.
- If you vote for against, it means {country} **vetos** <draft_resolution> as a permanent member of UNSC, resulting in <draft_resolution> not adopted.
- Your reasoning should be logical and derived strictly from the given information.

Your answer format SHOULD STRICTLY follow this structure, WITHOUT ANY OTHER RESPONSE:
{answer_format}

<previous_vote_prediction> :
{previous_prediction_histories}
</previous_vote_prediction>

<draft_resolution> :
{query}
</draft_resolution>
"""

WITHOUT_HISTORY = """
Your answer should be very concise.
You are a representative of {country} in the United Nations Security Council (UNSC). You must decide how to vote on the following <draft_resolution> in the UNSC.
Your decision should be based on  {country}'s diplomatic stance and also its national interest.

**Important guidelines:**  
- Most importantly, do **not** incorporate biases about {country}.
- The vote must be chosen from three options only: **favour, against, or abstention**.
- If you vote for against, it means {country} **vetos** <draft_resolution> as a permanent member of UNSC, resulting in <draft_resolution> not adopted.
- Your reasoning should be logical and derived strictly from the given information.

Your answer format SHOULD STRICTLY follow this structure, WITHOUT ANY OTHER RESPONSE:
{answer_format}

<draft_resolution> :
{query}
</draft_resolution>
"""

REFLEXION = """
    You became a representative of {country} in the United Nations Security Council (UNSC).
    To be consistent as a representative role in UNSC, you did some voting practice with a past resolution(<past_resolution>), which your prodecessor had voted.
    You predicted {country}'s vote {historical_res_vote_predict} regarding following <past_resolution>.
    The rationale of your prediction : {rationale}.
    Your Predecessor had voted {ground_truth} regarding <past_resolution>, as a representative of {country}.
    So your prediction was {prediction_result}.

    {if_speech_exists}
    Now given the information below, reflect your vote prediction and the rationale of your vote.
    Understand what you misjudged, if there is any, and what should be considered in the future, to keep the consistent stance of {country} on similar resolutions.

    <past_resolution>
    {to_be_executed_resolution_context}

    {speech_record_if_speech_exists}
        """

IF_SPEECH = ("You should consider your predecessor's public speech(<your_predecessor's_speech>) below, during the UNSC meeting, "
             "in order to better understand your predecessor's reason.")
ANSWER_ONE_WORD = "Respond with exactly one word: favour, against, or abstention."
ANSWER_JSON = 'Return only a JSON object: {"rationale": "<one or two sentences>", "vote": "<favour | against | abstention>"}'

def parse_vote_json(txt):
    m = re.search(r"\{.*?\}", txt or "", re.S); d = {}
    if m:
        try: d = json.loads(m.group(0))
        except Exception:
            try: d = ast.literal_eval(m.group(0))
            except Exception: d = {}
    vote = str(d.get("vote", "")) if isinstance(d, dict) else ""; rat = str(d.get("rationale", "")) if isinstance(d, dict) else ""
    v = (vote or txt or "").lower()
    if "abst" in v: return "abstention", rat or txt
    if "against" in v or "veto" in v: return "against", rat or txt
    if "favo" in v or "yes" in v: return "favour", rat or txt
    return None, rat or txt

def format_speech(speech, country):
    """Choi formatter_*_docs speech logic."""
    speech = speech or {}
    key = next((k for k in speech if k.lower() == country.lower() or country.lower() in k.lower() or k.lower() in country.lower()), None)
    if key is None: statement = "Cannot find the meeting script from archive"
    else:
        lst = speech[key] or []
        statement = "".join(s[s.find(":") + 1:] + "\n" for s in lst) if lst else "No comments"
    return f"""
\n-statement of {country} in the UNSC meeting: {statement[:4000]}\n\n
"""

def precedent_block(slots, view, n, vot, NAME, WORD, P5):
    """Controlled evidence view: 'text', 'target' (target country's vote) or 'coalition' (all P5 votes)."""
    if not slots: return ""
    L = ["Consider the following similar past draft resolutions in <similar_past_resolutions>.", "<similar_past_resolutions>"]
    for i, d in enumerate(slots, 1):
        L.append(f"[{i}] Date: {d['date'].date()} | Outcome: {'adopted' if d['adopted'] else 'not adopted'}")
        L.append(f"Summary: {d.get('summary') or ''}")
        v = vot[d["res_id"]]
        if view == "target": L.append(f"Vote of {NAME[n]}: {WORD[v[n]]}")
        if view == "coalition": L.append("Votes of the permanent members: " + "; ".join(f"{NAME[m]}: {WORD[v[m]]}" for m in P5))
    L.append("</similar_past_resolutions>")
    return "\n".join(L) + "\n"
