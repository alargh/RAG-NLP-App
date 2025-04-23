def get_text_from_files(files):
    text = ""
    for file in files:
        text += file.read().decode("utf-8")
    return text