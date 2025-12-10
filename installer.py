from sec_api import PdfGeneratorApi
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import json
import requests
import argparse

def get_filing_links(user_email, cik, form_type="10-K", year=None):
    headers = {"User-Agent": user_email}
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    data = requests.get(url, headers=headers).json()
    filings = data["filings"]["recent"]

    results = []
    for form, acc, doc, date in zip(filings["form"], filings["accessionNumber"],
                                    filings["primaryDocument"], filings["filingDate"]):
        if form == form_type and (year is None or date.startswith(str(year))):
            acc_no = acc.replace("-", "")
            link = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/{doc}"
            results.append((date, link))
            
    return results

def download_dataset(generator_key:str, output_dir:str, company_ticket_path:str, N:int = None):
    with open(company_ticket_path) as fp:
        cik_dict = json.load(fp)
        
    if N is not None:
        cik_dict = {k: v for k, v in list(cik_dict.items())[:N]}

    for cik in (cik_dict.items()):
        try:
            filing_10K_url = get_filing_links(cik[1]['cik_str'], "10-K", 2024)[0][1]
            print("10-K filings:", cik[1]['ticker'], filing_10K_url)
        except:
            print("Fail to retrieve ", cik[1]['ticker'])
            continue

        # download 10-K filing as PDF
        pdf_10K_filing = generator_key.get_pdf(filing_10K_url)
        file_name = filing_10K_url.split('/')[-1].split('.')[0]
        # save PDF of 10-K filing to disk
        with open(f"{output_dir}/SEC/{file_name}_10K.pdf", "wb") as file:
          file.write(pdf_10K_filing)

def split_and_embed(data_path:str, chunk_size:int, chunk_overlap:int, separator:str, model_name:str, device:str, database_path:str):
    
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size,
                                         chunk_overlap=chunk_overlap,
                                         separator=separator)
    source_docs = text_splitter.split_documents(documents)

    #Add company and year metadata based on the filename of the source
    for d in source_docs:
        metainfo = d.metadata['source'].split('/')[-1].split('_')
        d.metadata['company'] = metainfo[0].split('-')[0]
        d.metadata['year'] = int(metainfo[0].split('-')[1][:4])
        if len(metainfo) >= 4:
            d.metadata['Q-K'] = metainfo[1].split('.')[0]
        else:
            d.metadata['Q-K'] = "Annual"

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True} # set True to compute cosine simi
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    db = Chroma.from_documents(source_docs, embedding=hf, persist_directory=database_path)
    

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Parse arguments for the financial chatbot.")
    
    # Add an optional argument with a default value
    parser.add_argument('--generator-key', type=str, default='',
                        help='Specify your created PdfGeneratorApi')
    parser.add_argument('--filing-dir', type=str, default='./',
                       help='Specify SEC Filing Downloaded Documents.')
    parser.add_argument('--ticket-path', type=str, default='./company_tickers.json',
                       help='Specify the company ticket path')
    parser.add_argument('--N', type=int, default=500,
                       help='Filter the first N company records')
    parser.add_argument('--document-path', type=str, default=f'{filing_dir}/SEC/',
                       help='Specify the SEC document path')
    parser.add_argument('--chunk-size', type=int, default=7000,
                       help='Indicate document chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=700,
                       help='Specify chunk overlap')
    parser.add_argument('--separator', type=str, default="\n",
                       help='Specify the chunking separator')
    parser.add_argument('--encoder', type=str, default='BAAI/bge-small-en',
                       help='Specify the retriever model')
    parser.add_argument('--database-path', type=str, default='./db/')

    args = parser.parse_args()
    
    device = torch.device('cuda:0')

    pdfGeneratorApi = PdfGeneratorApi(args.generator_key)
    download_dataset(pdfGeneratorApi, args.filing_dir, args.ticket_path, args.N)
    split_and_embed(args.document_path, args.chunk_size, args.chunk_overlap, args.separator, args.encoder, device, args.database_path)