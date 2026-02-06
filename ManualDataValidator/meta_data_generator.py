import os 
import json
import shutil 
from google import genai
from dotenv import load_dotenv
import random 

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

API_KEY = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
]

client = genai.Client(api_key=random.choice(API_KEY))

def list_json(file_path):
    for i,   jsonfile in enumerate(os.listdir(file_path)):
        if jsonfile.endswith('.json'):
            print(f"{i+1} : {jsonfile}")
    print("\n")

def get_prompt(file_path: str) -> str:

    with open(file_path, "r") as f:
        system_prompt = f.read()
        f.close()
        return system_prompt
    
def get_content_from_file_to_exclude(file_wanna_exclude: list) -> str:  
    exclude_content = ""
    for file_name in file_wanna_exclude:
        try:
            with open(file_name, 'r') as f:
                content = f.read()
                exclude_content += content + "\n"
        except FileNotFoundError:
            print(f"File {file_name} not found. Skipping.")
    return exclude_content

def scanner(books_path: str, research_papers_path: str) -> list:
    """
    Scans a folder and returns a list of all files within it.
    """
    folder_path = [books_path, research_papers_path]
    book_list = []
    paper_list = []
    for folder in folder_path:
        for number, file in enumerate(os.listdir(folder)):
            if folder == books_path:
                book_list.append(file)
            else:
                paper_list.append(file)
    return book_list , paper_list

def json_generator(book_list: list, research_paper_list: list) -> json:
    """
    Take book list and return json about meta data.
    """
    list_json('./')
    till_day: int = int(input("Enter the till day numbers you want to exclude: ")) # 4 halyo vane 1,2,3,4 exclude garne 
    to_exclude_arr = [i for i in range(1, till_day + 1)]  #till_day = 4---> l =  [1,2,3,4]
    file_wanna_exclude = []
    if till_day == 0:
        #only for day 1 
        file_wanna_exclude = []
    else:
        file_wanna_exclude = [f'meta_data_day_{i}.json' for i in to_exclude_arr]
        exclude_content = get_content_from_file_to_exclude(file_wanna_exclude)


    prompt = get_prompt('system_prompt.txt')
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt.strip() + str(book_list) + str(research_paper_list) + str(exclude_content),
    )
    raw =  response.candidates[0].content.parts[0].text

    if not raw:
        raise ValueError("Model returned empty response")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print("Invalid JSON returned by model:")
        print(raw)
        raise e

    return json.dumps(parsed, indent=4)

def json_file_writer(meta_data: json, data_collection_day: int):
    """
    Write json meta data to a file.
    """
    with open(f'meta_data_day_{data_collection_day}.json', 'w') as f:
        f.write(meta_data)



if __name__ == "__main__":
    BOOK_PATH = 'D:\\DataSetForLLM\\Books'
    RESEARCH_PAPERS_PATH = 'D:\\DataSetForLLM\\ResearchPapers'
    data_collection_day = int(input("Enter data collection day: "))
    booko_list , paper_list = scanner(BOOK_PATH, RESEARCH_PAPERS_PATH)
    meta_data = json_generator(booko_list, paper_list)
    json_file_writer(meta_data, data_collection_day)    

