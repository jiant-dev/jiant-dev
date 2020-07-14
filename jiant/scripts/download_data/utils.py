import os
import tarfile
import urllib
import zipfile


def download_and_unzip(url, extract_location):
    """Downloads and unzips a file, and deletes the zip after"""
    _, file_name = os.path.split(url)
    zip_path = os.path.join(extract_location, file_name)
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zip_ref:
        zip_ref.extractall(extract_location)
    os.remove(zip_path)


def download_and_untar(url, extract_location):
    _, file_name = os.path.split(url)
    """Downloads and untars a file, and deletes the tar after"""
    tar_path = os.path.join(extract_location, file_name)
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_location)
    os.remove(tar_path)
