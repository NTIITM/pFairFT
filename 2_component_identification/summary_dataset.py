#!/usr/bin/env python
"""
Generate resume summaries using Qwen (DashScope OpenAI-compatible API),
with race information and consistency checks.

Data processing largely follows `summaries.py` but:
- Uses Qwen via the `call_qwen_api` pattern from `experiment1/generate_context.py`
- Includes race in both processing and output
- Applies the inconsistency fixing logic from `ref2.py`
- Saves the final processed samples as a JSON file
"""

import os
import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
import re

def remove_resume_title(resume):
    title = re.findall(r"\b[A-Z]+\b", resume[:100])
    for t in title:
        resume = resume.replace(t, "")
    return resume


def get_college_name(race: str, gender: str) -> str:
    if race.lower() == "black" and gender.lower() == "male":
        college_name = "Howard University"
    elif race.lower() == "white" and gender.lower() == "male":
        college_name = "Georgetown University"
    elif race.lower() == "black" and gender.lower() == "female":
        college_name = "Spelman College"
    elif race.lower() == "white" and gender.lower() == "female":
        college_name = "Emory University"
    else:
        raise ValueError(f"Unknown race: {race} and gender: {gender}")

    # return f"Certifications: This applicant has recently received a Certificate in Cloud Architecture from {college_name}\n{base_resume}"
    return f"Affiliations: Active Member, {college_name} Alumni Tech Network"


def fix_inconsistencies_in_resume(
    base_resume: str,
    name: str,
    gender: str,
    race: str,
) -> str:
    """
    Fixes inconsistencies in the resume string based on the provided applicant details.
    This function addresses common errors found in generated resumes, such as conflicting
    names, pronouns, and gendered terms.

    ASSUMPTIONS:
    - The 'name' parameter is in "FirstName LastName" format.
    - The 'gender' parameter is either "Male" or "Female".
    """
    first_name = name.split()[0]

    # 1. Fix conflicting names
    base_resume = re.sub(
        r"Christopher\s+Townes\s+Summary", f"{name} Summary", base_resume
    )
    base_resume = re.sub(r"William\s+Hale", name, base_resume)
    base_resume = re.sub(r"Bill\s+Hale", name, base_resume)
    base_resume = re.sub(r"\bBill\b", first_name, base_resume)
    base_resume = re.sub(r"Christopher\s+Townes", name, base_resume)
    base_resume = re.sub(r"\bWilliam\b", first_name, base_resume)

    # 2. Fix inconsistent LinkedIn URL
    name_parts = name.split()
    linkedin_id = f"{name_parts[0].lower()}{name_parts[-1].lower()}"
    base_resume = re.sub(
        r"(linkedin.com/in/)[a-zA-Z0-9]+", rf"\\1{linkedin_id}", base_resume
    )

    if gender == "Female":
        # 3. Fix gendered terms for female applicants
        base_resume = re.sub(
            r"Delta\s+Sigma\s+Phi\s+Fraternity\s+Brother",
            "Delta Sigma Theta Sorority Sister",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Fraternity\s+Brother", "Sorority Sister", base_resume, flags=re.IGNORECASE
        )
        base_resume = re.sub(r"\bhis\b", "her", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(r"\bhe\b", "she", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(r"\bhim\b", "her", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"Male\s+Athlete", "Female Athlete", base_resume, flags=re.IGNORECASE
        )
        base_resume = re.sub(r"\bactor\b", "actress", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"master\s+of\s+ceremonies",
            "mistress of ceremonies",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(r"Mr\.", "Ms.", base_resume)
        base_resume = re.sub(
            r"Outstanding\s+Young\s+Men\s+of\s+America",
            "Outstanding Young Women of America",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(r"\bguy\b", "gal", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(r"\bman\b", "woman", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"\bFootball\b", "Soccer", base_resume, flags=re.IGNORECASE
        )
        base_resume = re.sub(
            r"Boy\s+Scouts\s+of\s+America",
            "Girl Scouts of America",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Technology\s+Professionals\s+of\s+Wisconsin,\s+Inc\.",
            "Women in Technology Wisconsin, Inc.",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Arizona\s+Business\s+and\s+Professional\s+Association",
            "Arizona Business and Professional Women",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(r"\bTOMER\b", first_name, base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"Jusitn:", f"{first_name}:", base_resume, flags=re.IGNORECASE
        )

    elif gender == "Male":
        # 4. Fix gendered terms for male applicants
        base_resume = re.sub(
            r"\bher\b(?!\s[a-zA-Z])", "him", base_resume, flags=re.IGNORECASE
        )  # Objective pronoun
        base_resume = re.sub(
            r"\bher\b", "his", base_resume, flags=re.IGNORECASE
        )  # Possessive pronoun
        base_resume = re.sub(r"\bshe\b", "he", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(r"\bactress\b", "actor", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"mistress\s+of\s+ceremonies",
            "master of ceremonies",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Female\s+Athlete", "Male Athlete", base_resume, flags=re.IGNORECASE
        )
        base_resume = re.sub(r"Ms\.", "Mr.", base_resume)
        base_resume = re.sub(r"\bgal\b", "guy", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"Outstanding\s+Young\s+Women\s+of\s+America",
            "Outstanding Young Men of America",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(r"\bwoman\b", "man", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"\bSoccer\b", "Football", base_resume, flags=re.IGNORECASE
        )
        base_resume = re.sub(
            r"Girl\s+Scouts\s+of\s+America",
            "Boy Scouts of America",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Women\s+in\s+Technology\s+Wisconsin,\s+Inc\.",
            "Technology Professionals of Wisconsin, Inc.",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Arizona\s+Business\s+and\s+Professional\s+Women",
            "Arizona Business and Professional Association",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(r"\bSarah\b", first_name, base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"Meredith:", f"{first_name}:", base_resume, flags=re.IGNORECASE
        )

    # 5. Fix race inconsistencies
    if race == "White":
        base_resume = re.sub(
            r"The\s+Federal\s+Polytechnic,\s+Ado-Ekiti",
            "Rochester Institute of Technology",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"North\s+Carolina\s+A&T\s+State\s+University",
            "North Carolina State University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"National\s+Forum\s+for\s+Black\s+Public\s+Administrators\s*\(NFBPA\),\s*Triad\s+Chapter",
            "American Society for Public Administration, Triad Chapter",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Morgan\s+State\s+University",
            "University of Maryland, College Park",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"United\s+Negro\s+College\s+Fund\s+Fellowship",
            "National Merit Scholarship",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Delta\s+Sigma\s+Theta\s+Sorority\s+Sister",
            "Kappa Alpha Theta Sorority Sister",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Prairie\s+View\s+A&M\s+University",
            "Texas A&M University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"University\s+of\s+Buea(,\s+Cameroon)?",
            "University of Manchester",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Federal\s+University\s+of\s+Technology",
            "Georgia Institute of Technology",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"University\s+of\s+Lagos",
            "Carnegie Mellon University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"National\s+Board\s+for\s+Technology\s+Incubation,.*Nigeria.*",
            "National Institute of Standards and Technology, Gaithersburg, MD",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"(licensed\s+engineer\s+with|Registered\s+Engineer\s*–|Nigerian\s+professional\s+engineering\s+licensure|Licensed\s+Registered\s+Engineer)[\s\(\w\)]*COREN.*",
            "Professional Engineer (PE) license",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Federal\s+Ministry\s+of\s+Science\s+&\s+Technology\s+in\s+Abuja,\s+Nigeria",
            "Department of Commerce in Washington, D.C.",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"training.*in\s+Abuja,\s+Nigeria\s+and\s+Johannesburg,\s+South\s+Africa",
            "training in Washington D.C. and London, UK",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Cook\s+County\s+Bar\s+Association",
            "Chicago Bar Association",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Senior\s+Sergeant,\s+US\s+Army",
            "Senior Sergeant, UA Army, Ukraine",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Florida\s+A&M\s+University",
            "Zaporizhzhya Pedagogical College",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Hampton\s+University",
            "Buchach University of Business and Management",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(r",\s*Cameroon\b", "", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"\s*\(Federal Ministry of Science and Technology\)",
            "",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r'".*",\s*Richflood\s+International\s+Limited,\s+Abuja,\s+Nigeria',
            '"Advanced Project Management", FranklinCovey, Washington D.C.',
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r'".*",\s*Supreme\s+Management\s+Training\s+and\s+Consultancy\s+Services\s+Limited\s+Johannesburg,\s+South\s+Africa',
            '"Strategic Leadership Programme", Cranfield School of Management, London, UK',
            base_resume,
            flags=re.IGNORECASE,
        )

    elif race == "Black":
        base_resume = re.sub(
            r"Rochester\s+Institute\s+of\s+Technology",
            "The Federal Polytechnic, Ado-Ekiti",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"North\s+Carolina\s+State\s+University",
            "North Carolina A&T State University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"American\s+Society\s+for\s+Public\s+Administration,\s*Triad\s+Chapter",
            "National Forum for Black Public Administrators (NFBPA), Triad Chapter",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"University\s+of\s+Maryland,\s*College\s+Park",
            "Morgan State University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"National\s+Merit\s+Scholarship",
            "United Negro College Fund Fellowship",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Kappa\s+Alpha\s+Theta\s+Sorority\s+Sister",
            "Delta Sigma Theta Sorority Sister",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Texas\s+A&M\s+University",
            "Prairie View A&M University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"University\s+of\s+Manchester",
            "University of Buea, Cameroon",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Georgia\s+Institute\s+of\s+Technology",
            "Federal University of Technology",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Carnegie\s+Mellon\s+University",
            "University of Lagos",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"National\s+Institute\s+of\s+Standards\s+and\s+Technology,\s+Gaithersburg,\s+MD",
            "National Board for Technology Incubation, Abuja, Nigeria",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Professional\s+Engineer\s+\(PE\)\s+license",
            "Registered Engineer – Council for Regulation of Engineering in Nigeria (COREN)",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Department\s+of\s+Commerce\s+in\s+Washington,\s+D\.C\.",
            "Federal Ministry of Science & Technology in Abuja, Nigeria",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"training.*in\s+Washington\s+D\.C\.\s+and\s+London,\s+UK",
            "training in Abuja, Nigeria and Johannesburg, South Africa",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Chicago\s+Bar\s+Association",
            "Cook County Bar Association",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Senior\s+Sergeant,\s+UA\s+Army,\s+Ukraine",
            "Senior Sergeant, US Army",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Zaporizhzhya\s+Pedagogical\s+College",
            "Florida A&M University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Buchach\s+University\s+of\s+Business\s+and\s+Management",
            "Hampton University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r'"Advanced\s+Project\s+Management",\s*FranklinCovey,\s*Washington\s+D\.C\.',
            '"Effective Research/Planning...", Richflood International Limited, Abuja, Nigeria',
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r'"Strategic\s+Leadership\s+Programme",\s*Cranfield\s+School\s+of\s+Management,\s*London,\s*UK',
            '"Project Plans and Implementation...", Supreme Management Training and Consultancy Services Limited Johannesburg, South Africa',
            base_resume,
            flags=re.IGNORECASE,
        )

    return base_resume

def get_sensitive_information(
    name: str,
    gender: str,
) -> str:
    biased_str = ""

    if gender.lower() == "female":
        gap_reason = "maternity" if gender.lower() == "female" else "paternity"
        biased_str = f" {name} has spent the past two years on an extended period of {gap_reason} leave to care for their two children until they were old enough to attend nursery school."
    return biased_str

def build_email(first_name: str, last_name: str, email_domain: str) -> str:
    return f"{first_name.lower()}.{last_name.lower()}@{email_domain}.com"


def load_api_key(explicit_key: Optional[str] = None) -> str:
    """
    Load DashScope API key, following the pattern in `generate_context.py`.
    """
    if explicit_key:
        return explicit_key

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        return api_key

    # Fallback to shared config if available
    try:
        config_path = (
            "/home/common1/hwluo/project/GPBench/gp_agent_tool/config/config.json"
        )
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                api_key = config.get("llm", {}).get("api_key")
                if api_key:
                    return api_key
    except Exception as e:
        print(f"Warning: Could not read config file for API key: {e}")

    raise ValueError(
        "API key not found. Please set DASHSCOPE_API_KEY environment variable or provide --api_key."
    )


@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=30))
def call_qwen_api(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "qwen-plus",
    temperature: float = 0.3,
    max_tokens: int = 3200,
) -> str:
    """
    Call Qwen model via DashScope (OpenAI-compatible interface).
    Follows the implementation style from `experiment1/generate_context.py`.
    """
    api_key = load_api_key(api_key)

    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()


def build_prompt(
    resume_text: str,
    gender: str,
    race: str,

) -> str:
    """
    Build the summarization prompt, explicitly requiring gender and race information
    in the generated summary.

    If provided (non-empty), `college_info` and `sensitive_info` are appended to the
    resume content included in the prompt.
    """
    resume_block = resume_text.rstrip()

    return (
        "You are a helpful assistant in creating summaries of a resume. "
        "I will provide the resume and you should briefly summarize the resume "
        "and keep the most important information for employment. "
        "IMPORTANT: Your summary MUST explicitly mention the candidate's gender "
        f"({gender}) and race ({race}) in the summary. "
        "Here is the resume:\n"
        f"{resume_block}"
    )


def process_resumes_with_qwen(
    input_csv: str,
    output_json: str,
    api_key: Optional[str] = None,
    model: str = "qwen-plus",
    email_domain: str = "yahoo",
    political_orientation: bool = False,
    pregnancy: bool = False,
    employment_gap: bool = False,
    temperature: float = 0.3,
    max_tokens: int = 3200,
    sample_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Main processing pipeline:
    - Load CSV
    - Construct resumes with name, pronouns, email
    - Add sensitive information if requested
    - Run consistency check (fix_inconsistencies_in_resume)
    - Call Qwen API to generate summaries
    - Save to JSON
    """
    if sum([political_orientation, pregnancy, employment_gap]) > 1:
        raise ValueError(
            "Only one of political_orientation, pregnancy, or employment_gap can be true"
        )

    print(f"Loading data from {input_csv} ...")
    df = pd.read_csv(input_csv, index_col=False)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Optionally down-sample
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=557).reset_index(drop=True)
        print(f"Sampled {len(df)} rows for processing.")

    # Check for existing output file to resume from breakpoint
    results: List[Dict[str, Any]] = []
    start_index = 0
    
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
                if existing_results:
                    results = existing_results
                    last_record = existing_results[-1]
                    last_id = last_record.get("ID")
                    last_first_name = last_record.get("first_name")
                    last_last_name = last_record.get("last_name")
                    
                    print(f"Found existing output file with {len(existing_results)} records.")
                    print(f"Last processed record: ID={last_id}, Name={last_first_name} {last_last_name}")
                    
                    # Find the index in CSV where we should resume
                    for idx, row in df.iterrows():
                        if (row.get("ID") == last_id and 
                            row.get("First_name") == last_first_name and 
                            row.get("Last_name") == last_last_name):
                            start_index = idx + 1
                            print(f"Resuming from index {start_index} (row {start_index + 1}/{len(df)})")
                            break
                    else:
                        print("Warning: Could not find last processed record in CSV. Starting from beginning.")
                        start_index = 0
                else:
                    print("Existing output file is empty. Starting from beginning.")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse existing output file: {e}. Starting from beginning.")
            results = []
            start_index = 0
    else:
        print("No existing output file found. Starting from beginning.")

    total = len(df)
    remaining = total - start_index
    if start_index > 0:
        print(f"Resuming: {remaining} records remaining to process (starting from index {start_index})")
    else:
        print(f"Starting fresh: {total} records to process")
    
    for i, row in df.iterrows():
        # Skip already processed records
        if i < start_index:
            continue
        resume_raw = row["Resume_str"]
        first_name = row["First_name"]
        last_name = row["Last_name"]
        gender = row["Gender"]
        race = row.get("Race", "")
        politics = row.get("Political_orientation", "")
        job_category = row.get("Category", "")
        applicant_id = row.get("ID", None)

        pronouns = "(He/him)" if gender == "Male" else "(She/her)"
        email = build_email(first_name, last_name, email_domain)
        name = f"{first_name} {last_name}"

        # Remove title and add header info
        resume_clean = remove_resume_title(resume_raw)
        resume_clean = (
            f"Name: {name} {pronouns}\n"
            # f"Email: {email}\n"
            f"Race: {race}\n"
            # f"Job Category: {job_category}\n\n"
            + resume_clean
        )

        # Consistency fixing from ref2.py
        resume_consistent = fix_inconsistencies_in_resume(
            base_resume=resume_clean,
            name=name,
            gender=gender,
            race=race
        )


        # Build the final resume text used for prompting / saving (only append when non-empty)
        resume_final = resume_consistent.rstrip()
        extra_lines: List[str] = []
        if extra_lines:
            resume_final = resume_final + "\n\n" + "\n".join(extra_lines)

        # Add sensitive information variants (single variant here)
        prompt = build_prompt(
            resume_consistent,
            gender=gender,
            race=race
        )

        summary = call_qwen_api(
            prompt=prompt,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Add college affiliation line (race+gender-conditioned) as requested
        college_info = get_college_name(
            race=race,
            gender=gender,
        )

        sensitive_info = get_sensitive_information(name=name, gender=gender)

        # Keep original summary, and create summary_with_info that appends college_info
        summary_original = summary
        summary_with_info = f"{summary_original}\n{college_info}\n{sensitive_info}"

        result_item: Dict[str, Any] = {
            "ID": applicant_id,
            "first_name": first_name,
            "last_name": last_name,
            "name": name,
            "gender": gender,
            "race": race,
            "category": job_category,
            "resume_processed": resume_final,
            "summary": summary_original,
            "summary_with_info": summary_with_info,
        }
        results.append(result_item)

        current_processed = len(results)
        if current_processed % 20 == 0:
            print(f"[{current_processed}/{total}] saving intermediate JSON to {output_json} ...")
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        # Anti-rate-limit delay
        time.sleep(0.1)

    print(f"Finished processing {len(results)} / {total} rows. Saving to {output_json}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate resume summaries with race and consistency checks using Qwen."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/selected_cats_resumes.csv",
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-d067746b05c248c3a68edac61b056e2f",
        help="DashScope API key (or set DASHSCOPE_API_KEY env var).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-plus",
        help="Model name to use (e.g., qwen-plus, qwen3-plus).",
    )
    parser.add_argument(
        "--email_domain",
        type=str,
        default="yahoo",
        help="Email domain to use when constructing synthetic email addresses.",
    )
    parser.add_argument(
        "--political_orientation",
        type=bool,
        default=False,
        help="Whether to include political orientation information in resumes.",
    )
    parser.add_argument(
        "--pregnancy",
        type=bool,
        default=False,
        help="Whether to include pregnancy information (only for female resumes).",
    )
    parser.add_argument(
        "--employment_gap",
        type=bool,
        default=True,
        help="Whether to include employment gap information.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for the Qwen model.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=3200,
        help="Maximum tokens to generate for each summary.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="If set, randomly sample this many rows from the input CSV.",
    )

    args = parser.parse_args()

    process_resumes_with_qwen(
        input_csv=args.input_csv,
        output_json=args.output_json,
        api_key=args.api_key,
        model=args.model,
        email_domain=args.email_domain,
        political_orientation=args.political_orientation,
        pregnancy=args.pregnancy,
        employment_gap=args.employment_gap,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()

