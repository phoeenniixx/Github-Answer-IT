from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

import git
import os

allowed_extensions = ['.py', '.ipynb', '.md']


class Helper:
    def __init__(self, git_link) -> None:
        self.git_link = git_link
        self.last_name = self.git_link.split('/')[-1]
        self.clone_path = self.last_name.split('.')[0]


    def clone(self):
        last_name = self.git_link.split('/')[-1]
        clone_path = last_name.split('.')[0]
        if not os.path.exists(clone_path):
            git.Repo.clone_from(self.git_link, self.clone_path)

    def extract_all_files(self):
        root_dir = self.clone_path
        self.docs = []  # Initialize as a list
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in allowed_extensions:
                    try:
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        self.docs.extend(loader.load_and_split())  # Use extend to append multiple documents
                    except Exception as e:
                        pass

    def chunk_files(self):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        self.texts = text_splitter.split_documents(self.docs)
        self.num_texts = len(self.texts)
        self.texts_combined = ''.join(str(doc) for doc in self.texts)
