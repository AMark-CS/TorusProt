import pandas as pd


def rename_processed_paths(file: str, str_to_replace: str, replacement: str) -> None:
    """
    Rename the paths in the specified column of the DataFrame..

    Args:
        df (pd.DataFrame): The DataFrame containing the paths.
    """
    df = pd.read_csv(file)
    df["processed_path"] = df["processed_path"].str.replace(str_to_replace, replacement)
    df.to_csv(file, index=False)


if __name__ == "__main__":
    str_to_replace = "/home/stas/code_projects/"
    replacement = "/workspace/"
    file = "metadata_all.csv"
    rename_processed_paths(file, str_to_replace, replacement)
