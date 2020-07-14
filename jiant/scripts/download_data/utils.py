import urllib
import zipfile
import os


def download_and_unzip(url, extract_location, temp_zip_file_name="__temp.zip"):
    """Downloads and unzips a file, and deletes the zip after"""
    zip_path = os.path.join(extract_location, temp_zip_file_name)
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zip_ref:
        zip_ref.extractall(extract_location)
    os.remove(zip_path)
