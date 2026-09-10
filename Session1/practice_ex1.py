import numpy as np
from PIL import Image
from session1 import load_data, show_samples, IMG_DIR

#Question 1!
def q1_most_common_malignant_subclass(df):
    malignant = df[df["diagnosis_1"] == "Malignant"]
    counts = malignant["diagnosis_2"].value_counts()

    print(counts.head(10))           
    print(f"\nAnswer Q1: most common Malignant subcategory is "
          f"'{counts.index[0]}' with {counts.iloc[0]} images "
          f"({counts.iloc[0] / len(malignant):.1%} of Malignant).")


# Question 2!
def q2_average_image_size(df, n=200):
    widths, heights = [], []

    for _, row in df.sample(n, random_state=0).iterrows():
        with Image.open(IMG_DIR / f'{row["isic_id"]}.jpg') as im:
            w, h = im.size            # PIL reports size as (width, height)
        widths.append(w)
        heights.append(h)

    print(f"Answer Q2 ({n}-image sample):")
    print("  avg width :", np.mean(widths))
    print("  avg height:", np.mean(heights))
    print("  width  min/max:", min(widths), "/", max(widths))
    print("  height min/max:", min(heights), "/", max(heights))


# Question 3!
def q3_compare_two_diagnoses(df):
    show_samples(df, IMG_DIR, n=6, diagnosis="Malignant", seed=0)
    show_samples(df, IMG_DIR, n=6, diagnosis="Benign", seed=0)

    observations = """
    Malignant vs Benign - what I noticed:
      - Malignant lesions tend to be redder, whereas benign lesions are more of a dark brown color
      - Malignant lesions are more irregular in shape. Benign lesions are more regular and uniform in shape.
    """
    print(observations)


# Question 4!
def q4_check_one_lesion(df, lesion_id=None):

    if lesion_id is None:
        lesion_id = df["lesion_id"].iloc[0]

    pair = df[df["lesion_id"] == lesion_id]          
    print(f"Lesion {lesion_id}:")
    print(pair[["isic_id", "image_type"]].to_string(index=False))

    types = sorted(pair["image_type"])
    expected = ["clinical: close-up", "dermoscopic"]
    ok = (len(pair) == 2) and (types == expected)
    print("Answer Q4:", "CONFIRMED" if ok else "NOT as expected", "->", types)

    every_lesion_ok = (
        df.groupby("lesion_id")["image_type"]
          .apply(lambda s: sorted(s) == expected)
          .all()
    )
    print("Bonus - every lesion has exactly one of each image_type:", every_lesion_ok)


if __name__ == "__main__":
    df, gt = load_data()          

    print("\nQuestion 1")
    q1_most_common_malignant_subclass(df)

    print("\nQuestion 2")
    q2_average_image_size(df)

    print("\nQuestion 3")
    q3_compare_two_diagnoses(df)     

    print("\nQuestion 4")
    q4_check_one_lesion(df)