from groq import Groq
import os
from dotenv import load_dotenv
import time

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

print("GROQ KEY FOUND:", api_key is not None)

client = Groq(api_key=api_key)


def ask_agent(system_prompt, user_query):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print("GROQ ERROR:", str(e))
        raise


def parse_research(research_text):
    papers = []
    for i in range(1, 4):
        try:
            title = research_text.split(f"PAPER{i}_TITLE:")[1].split("\n")[0].strip()
            authors = research_text.split(f"PAPER{i}_AUTHORS:")[1].split("\n")[0].strip()
            source = research_text.split(f"PAPER{i}_SOURCE:")[1].split("\n")[0].strip()
            url = research_text.split(f"PAPER{i}_URL:")[1].split("\n")[0].strip()
            summary = research_text.split(f"PAPER{i}_SUMMARY:")[1].split("\n")[0].strip()
            papers.append({
                "title": title,
                "authors": authors,
                "source": source,
                "url": url,
                "summary": summary,
            })
        except Exception:
            pass
    return papers

def parse_risk_level(final_text):
    risk_map = {
        "high": {"label": "High", "width": "85%", "color": "#ffb4ab"},
        "moderate": {"label": "Moderate", "width": "50%", "color": "#4edea3"},
        "low": {"label": "Low", "width": "20%", "color": "#adc6ff"}
    }
    
    first_line = final_text.strip().split("\n")[0].lower()
    
    for key in risk_map:
        if key in first_line:
            # Strip the RISK_LEVEL line out of the displayed text
            clean_text = final_text.split("\n", 1)[1].strip() if "\n" in final_text else final_text
            return risk_map[key], clean_text
        
    return risk_map["moderate"], final_text

def run_doctoraai(user_query):

    print("Agent A thinking...")
    diagnosis = ask_agent(
        """You are a medical diagnosis specialist. 
        Analyze the given symptoms carefully.watch the symptomps as well as the duration of them
        List possible conditions from most to least likely.
        Always remind the user to consult a real doctor.
        Keep response clear and simple.""",
        user_query
    )
    time.sleep(5)  # wait 5 seconds

    print("Agent B thinking...")
    treatment = ask_agent(
        """You are a treatment and medication expert.
        Suggest general treatment approaches for given symptoms.
        Never recommend specific drug dosages.
        Always remind the user to consult a real doctor.""",
        user_query
    )
    time.sleep(5)  # wait 5 seconds

    print("Agent C thinking...")
    research = ask_agent(
        """You are a medical research assistant.
        Find 2-3 relevant research papers for the given symptoms.
        
        Return ONLY in this exact format, nothing else:
        
        PAPER1_TITLE: [title here]
        PAPER1_AUTHORS: [author here]
        PAPER1_SOURCE: [journal name]
        PAPER1_URL: https://pubmed.ncbi.nlm.nih.gov/?term=[relevant+search+terms]
        PAPER1_SUMMARY: [one line summary]
        
        PAPER2_TITLE: [title here]
        PAPER2_AUTHORS: [author here] 
        PAPER2_SOURCE: [journal name]
        PAPER2_URL: https://pubmed.ncbi.nlm.nih.gov/?term=[relevant+search+terms]
        PAPER2_SUMMARY: [one line summary]
        
        PAPER3_TITLE: [title here]
        PAPER3_AUTHORS: [author here]
        PAPER3_SOURCE: [journal name]
        PAPER3_URL: https://pubmed.ncbi.nlm.nih.gov/?term=[relevant+search+terms]
        PAPER3_SUMMARY: [one line summary]
        
        Use real PubMed search URLs with relevant medical terms.""",
        user_query
    )
    print("Validator synthesizing...")
    final = ask_agent(
            f"""You are a medical validation agent.
            Three specialist AI agents analyzed this query: "{user_query}"

            Agent A (Diagnosis) said:
            {diagnosis}

            Agent B (Treatment) said:
            {treatment}

            Agent C (Research) said:
            {research}

            Your job:
            1. Synthesize into a SHORT, structured response (max 200 words)
            2. Use this exact format:

            RISK_LEVEL: [Low/Moderate/High]

            **Likely Condition:** [2-3 conditions max]

            **Key Observations:** [2-3 bullet points only]

            **Recommended Actions:** [3-4 bullet points only]

            ⚠️ **Disclaimer:** Always consult a licensed medical professional.

            Risk level guide:
            - High: emergency symptoms, needs immediate medical attention
            - Moderate: should see a doctor soon, not urgent but concerning
            - Low: manageable at home, routine monitoring is enough

            Be concise. No long paragraphs. The RISK_LEVEL line must be the very first line.""",
            ""
        )

    return diagnosis, treatment, research, final
SPECIALIST_MAP = {
"headache": "Neurologist",
"migraine": "Neurologist",
"fever": "General Physician",
"cough": "Pulmonologist",
"breathing": "Pulmonologist",
"skin": "Dermatologist",
"rash": "Dermatologist",
"heart": "Cardiologist",
"chest pain": "Cardiologist",
"stomach": "Gastroenterologist",
"anxiety": "Psychiatrist",
"stress": "Psychiatrist",
"joint pain": "Orthopedic",
"bone": "Orthopedic"
}

def get_specialist(query):
    query = query.lower()

    for keyword, specialist in SPECIALIST_MAP.items():
        if keyword in query:
            return specialist

    return "General Physician"
def parse_lab_analysis(text):

    result = {
        "summary": "",
        "abnormal_count": 0,
        "values": [],
        "concerns": "",
        "specialist": "General Physician",
        "urgency": "Low"
    }

    try:

        lines = text.split("\n")

        for line in lines:

            if line.startswith("SUMMARY:"):
                result["summary"] = line.replace("SUMMARY:", "").strip()

            elif line.startswith("ABNORMAL_COUNT:"):
                result["abnormal_count"] = int(
                    line.replace("ABNORMAL_COUNT:", "").strip()
                )

            elif line.startswith("CONCERNS:"):
                result["concerns"] = line.replace("CONCERNS:", "").strip()

            elif line.startswith("SPECIALIST:"):
                result["specialist"] = line.replace("SPECIALIST:", "").strip()

            elif line.startswith("URGENCY:"):
                result["urgency"] = line.replace("URGENCY:", "").strip()

    except Exception as e:
        print(e)

    return result

if __name__ == "__main__":
    result = run_doctoraai(
        "I have a headache and fever since 2 days"
    )
    print(result)


