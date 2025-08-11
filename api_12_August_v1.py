import streamlit as st
import requests
from datetime import datetime
import concurrent.futures
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Configuration ===
TOGETHER_MODEL_LIST = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "meta-llama/Llama-Vision-Free",
    "lgai/exaone-deep-32b",
    "lgai/exaone-3-5-32b-instruct",
    "black-forest-labs/FLUX.1-schnell-Free",
    "arcee-ai/AFM-4.5B",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    "Best Champion Model"
]

LLM_PROVIDERS = ["Together AI"]
TEST_LEVELS = ["Unit", "Integration", "E2E"]
SCRIPT_LANGUAGES = ["Python", "Java", "JavaScript", "C#"]
OUTPUT_FORMATS = ["Code Only", "Code with Comments"]
TOGETHER_AI_API_KEY = "6e371349524ad0f7f7d9a15eb13c36dfd21d5caff4c96e3ad3515ad2aceeec54"

# === Session State Setup ===
if "generated_tests" not in st.session_state:
    st.session_state.generated_tests = {"api": [], "ui": []}
if "test_results_history" not in st.session_state:
    st.session_state.test_results_history = []
if "active_project" not in st.session_state:
    st.session_state.active_project = "Default"
if "projects" not in st.session_state:
    st.session_state.projects = {"Default": {}}

# === Champion Model Functions ===
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def process_in_chunks(text, max_chunk_size=1500):
    """Split text into chunks to avoid token limits"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        if current_size + len(word) + 1 <= max_chunk_size:
            current_chunk.append(word)
            current_size += len(word) + 1
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_champion_output(api_docs, test_case_count, api_key, test_level):
    semantic_model = load_semantic_model()
    
    def run_model(model, chunk=None):
        try:
            prompt = f"Generate {test_case_count} {test_level} test cases with complete details:\n{chunk if chunk else api_docs}"
            
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }],
                    "max_tokens": 4096  # Increased token limit
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            st.warning(f"Model {model} had an issue: {str(e)}")
            return None

    # Process in chunks if content is large
    chunks = process_in_chunks(api_docs)
    all_results = []
    
    # Exclude "Best Champion Model" from the models to run
    models_to_run = [model for model in TOGETHER_MODEL_LIST if model != "Best Champion Model"]
    
    with st.spinner(f"Running {len(models_to_run)} models in parallel..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for chunk in chunks:
                futures = {executor.submit(run_model, model, chunk): model for model in models_to_run}
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        all_results.append(result)

    # Combine and deduplicate results
    all_cases = []
    for res in all_results:
        all_cases.extend([c.strip() for c in res.split('\n\n') if c.strip()])
    
    # Semantic deduplication
    if all_cases:
        embeddings = semantic_model.encode(all_cases)
        similarity_matrix = cosine_similarity(embeddings)
        
        unique_cases = []
        used = set()
        for i in range(len(all_cases)):
            if i not in used:
                similar = np.where(similarity_matrix[i] > 0.8)[0]
                best = max([all_cases[idx] for idx in similar], key=len)
                unique_cases.append(best)
                used.update(similar)
        
        # Re-number all test cases sequentially
        formatted_cases = []
        for idx, case in enumerate(unique_cases[:test_case_count], 1):
            # Standardize numbering format
            if not case.strip().startswith(('**', str(idx))):
                case = f"{idx}. {case}"
            elif case.startswith('**') and not case.startswith(f"**{idx}"):
                case = f"**{idx}**" + case[case.index('**')+2:]
            formatted_cases.append(case)
        
        return "\n\n".join(formatted_cases)
    return "No test cases could be generated from the models."

# ------- Project Management UI -------
def project_management_ui():
    st.sidebar.header("Project Management")
    projects = list(st.session_state.projects.keys())
    if st.session_state.active_project not in projects:
        st.session_state.active_project = projects[0] if projects else None
    selected = st.sidebar.selectbox("Active Project", projects, index=projects.index(st.session_state.active_project))
    st.session_state.active_project = selected

    new_proj_key = "new_project_name_input"
    if st.sidebar.button("➕ New Project"):
        st.session_state[new_proj_key] = ""

    if new_proj_key in st.session_state and st.session_state[new_proj_key] == "":
        new_proj = st.sidebar.text_input("New Project Name", key=new_proj_key).strip()
        if new_proj and new_proj not in projects:
            st.session_state.projects[new_proj] = {}
            st.session_state.active_project = new_proj
            st.experimental_rerun()
            
# ------- JIRA Story Fetching Helper -------
def fetch_jira_story(jira_url, issue_key, username, token):
    try:
        url = f"{jira_url}/rest/api/2/issue/{issue_key}"
        headers = {"Authorization": f"Basic {token}"} if token else {}
        resp = requests.get(url, headers=headers, timeout=7)
        if resp.status_code == 200:
            return resp.json()["fields"]["summary"]
        else:
            return f"Error: {resp.text}"
    except Exception as e:
        return f"Error: {e}"
    
# ------- Get test cases from LLM -------
def get_test_cases_from_llm(user_story, test_case_count, api_key, model, test_level):
    if model == "Best Champion Model":
        return generate_champion_output(user_story, test_case_count, api_key, test_level)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Generate exactly {test_case_count} comprehensive {test_level.lower()} test cases with complete details. "
                    "Number each test case sequentially starting from 1. "
                    "Each test case must include:\n"
                    "1. Clear test case number (e.g., '1. Test Case Title')\n"
                    "2. Action/Endpoint\n"
                    "3. Input parameters\n"
                    "4. Expected result/status code\n"
                    "5. Any special conditions\n"
                    "Ensure all test cases are complete and not truncated."
                )
            },
            {
                "role": "user",
                "content": user_story
            }
        ],
        "max_tokens": 4096,  # Increased token limit
        "temperature": 0.5
    }
    endpoint = "https://api.together.xyz/v1/chat/completions"
    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
        
def generate_full_test_script_from_llm(tests, api_key, model, script_language, test_level, output_format):
    if output_format == "Code with Comments":
        prompt = (
            f"Convert these test cases into a properly formatted {script_language} test script with comments:\n"
            f"{tests}\n"
            "Include:\n"
            "1. Setup and teardown methods\n"
            "2. Test methods for each scenario\n"
            "3. Assertions for success criteria\n"
            "4. Proper code formatting with appropriate comments\n"
            "5. All necessary imports and dependencies\n"
            "Return ONLY the code with no additional text or Markdown formatting."
        )
    else:
        prompt = (
            f"Convert these test cases into a properly formatted {script_language} test script:\n"
            f"{tests}\n"
            "Include:\n"
            "1. Setup and teardown methods\n"
            "2. Test methods for each scenario\n"
            "3. Assertions for success criteria\n"
            "4. Proper code formatting without comments\n"
            "5. All necessary imports and dependencies\n"
            "Return ONLY the code with no additional text or Markdown formatting."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model if model != "Best Champion Model" else TOGETHER_MODEL_LIST[0],
        "messages": [
            {
                "role": "system",
                "content": f"You are a QA engineer creating clean {script_language} test scripts."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 4096,  # Increased token limit
        "temperature": 0.3
    }
    endpoint = "https://api.together.xyz/v1/chat/completions"
    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def main():
    st.set_page_config(page_title="AI-Powered Test Case Generation", layout="wide")
    st.markdown("<h1>AI-Powered Test Case Generation</h1>", unsafe_allow_html=True)
    
    project_management_ui()

    if not st.session_state.active_project:
        st.info("Please create or select a project to begin.")
        st.stop()

    st.success(f"🚀 Active Project: **{st.session_state.active_project}**")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Test Generation", "⚙️ Script Generation", "🚀 Execution", "📊 Reports"]
    )
    
    with tab1:
        st.header("1️⃣ Input: API Documentation or Requirements")
        uploaded = st.file_uploader("Upload Document (.txt, .md, .json)", type=["txt", "md", "json"])
        api_docs = ""

        if uploaded:
            try:
                api_docs = uploaded.read().decode("utf-8")
                st.text_area("Parsed Content", api_docs, height=200)
            except Exception as e:
                st.error(f"Read error: {e}")

        api_docs = st.text_area("Edit API Documentation", value=api_docs, height=200)

        st.subheader("Or Fetch from JIRA")
        proj = st.session_state.projects[st.session_state.active_project]
        jira_url = st.text_input("JIRA URL", value=proj.get("jira", {}).get("url", ""))
        issue_key = st.text_input("Issue Key (e.g., PROJ-123)")
        if st.button("Fetch from JIRA"):
            if jira_url and issue_key:
                token = proj.get("jira", {}).get("token")
                username = proj.get("jira", {}).get("user")
                story = fetch_jira_story(jira_url, issue_key, username, token)
                api_docs = story
                st.text_area("Fetched Content", story, height=200)

        if not api_docs:
            st.stop()

        st.header("2️⃣ Generate Test Cases")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Using pre-configured Together AI API key")
            model = st.selectbox("Select Model", TOGETHER_MODEL_LIST)

        with col2:
            test_level = st.selectbox("Test Level", TEST_LEVELS)
            test_case_count = st.slider("Number of Test Cases", min_value=5, max_value=50, value=20, step=1)

        if st.button("🧠 Generate Comprehensive Test Cases"):
            with st.spinner("Generating detailed test cases..."):
                try:
                    test_cases = get_test_cases_from_llm(
                        api_docs,
                        test_case_count,
                        TOGETHER_AI_API_KEY,
                        model=model,
                        test_level=test_level,
                    )
                    
                    # Store in session state
                    st.session_state.generated_tests["api"] = test_cases
                    st.session_state.llm_model = model
                    
                    # Display results
                    st.success("✅ Test cases generated!")
                    
                    # Show in expandable sections
                    with st.expander("View Generated Test Cases", expanded=True):
                        st.text_area("Test Cases", value=test_cases, height=500, label_visibility="collapsed")
                    
                    # Download buttons
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download as Text",
                            test_cases,
                            file_name=f"test_cases_{timestamp}.txt",
                            mime="text/plain"
                        )
                    with col2:
                        st.download_button(
                            "📥 Download as Word",
                            test_cases,
                            file_name=f"test_cases_{timestamp}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        
                except Exception as e:
                    st.error(f"Error generating test cases: {str(e)}")

    # -------- TAB 2: Script Generation --------
    with tab2:
        st.header("Generate Automation Scripts")
        if not st.session_state.get("generated_tests", {}).get("api"):
            st.warning("No test cases generated yet. Go to Tab 1.")
            st.stop()

        script_language = st.selectbox("Select Script Language", SCRIPT_LANGUAGES)
        output_format = st.selectbox("Select Output Format", OUTPUT_FORMATS)

        selected_model = st.session_state.get("llm_model")
        test_cases = st.session_state.generated_tests["api"]

        if st.button("Generate Test Script"):
            with st.spinner("Generating test script..."):
                try:
                    test_script = generate_full_test_script_from_llm(
                        tests=test_cases,
                        api_key=TOGETHER_AI_API_KEY,
                        model=selected_model,
                        script_language=script_language,
                        test_level=test_level,
                        output_format=output_format
                    )
                    
                    # Display the generated script
                    st.code(test_script, language=script_language.lower())
                    
                    # Download
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ext = script_language.lower()
                    st.download_button(
                        f"📥 Download Script (.{ext})",
                        test_script,
                        file_name=f"test_script_{timestamp}.{ext}",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Script generation failed: {str(e)}")

    # -------- TAB 3: Execution --------
    with tab3:
        st.header("Test Execution")
        st.info("Test execution functionality coming soon!")

    # -------- TAB 4: Reports --------
    with tab4:
        st.header("Test Reports")
        st.info("Reporting functionality coming soon!")

if __name__ == "__main__":
    main()