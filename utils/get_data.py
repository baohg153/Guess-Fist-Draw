from roboflow import Roboflow
import dotenv
import os
import shutil
from pathlib import Path

# dotenv.load_dotenv()

# def get_fist_dataset():
#     rf = Roboflow(api_key=os.getenv("API_KEY"))
#     project = rf.workspace("first-sixpi").project("fist-project")
#     version = project.version(3)

#     dataset = version.download(model_format="yolov7")

#     # src_dir = Path("fist-project-3")
#     # dest_dir = Path("data/fist/")
#     # for item in src_dir.iterdir():
#     #     shutil.move(str(item), dest_dir)

#     # os.rmdir("fist-project-3")
      
# get_fist_dataset()