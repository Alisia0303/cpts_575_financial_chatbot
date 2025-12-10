# cpts_575_financial_chatbot

# How to Install

Follow these steps to set up the project:

### 1. Clone the project:
git clone <repo-url>

### 2. **Create the virtual environment using Conda:**
Refer to our requirement.txt

### 3. **Run the installer:**
This step downloads SEC 10K fillings into `SEC` folder, then embeds records into Chroma database which is one of the vector store databases in LangChain. Because this step is time consuming when your computer doesn't have a gpu, I saved these data into `db` folder. Therefore, you can jump into the next step. However, if you would like to rerun or increase the number of embedding records, feel free to do it by following command line:

`python3 installer.py`