import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two result files and print the differences."
    )
    parser.add_argument("file1", type=str, help="Path to the first result file.")
    parser.add_argument("file2", type=str, help="Path to the second result file.")
    return parser.parse_args()


def compare_results(file1, file2):
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    if df1.shape != df2.shape:
        print("The files have different shapes.")
        return

    delta_polis = df1["polis"] - df2["polis"]
    delta_iou = df1["iou"] - df2["iou"]
    delta_ciou = df1["ciou"] - df2["ciou"]
    delta_mta = df1["mta"] - df2["mta"]

    # create a new DataFrame to hold the differences
    delta_df = pd.DataFrame({
        "image_id": df1["image_id"],
        "instance_id": df1["instance_id"],
        "#vertices": df1["#vertices"],
        "area": df1["area"],
        "size": df1["size"],
        "delta_polis": delta_polis,
        "delta_iou": delta_iou,
        "delta_ciou": delta_ciou,
        "delta_mta": delta_mta
    })

    return delta_df


def plot_histograms(delta_df):
    import matplotlib.pyplot as plt

    metrics = ["delta_polis", "delta_iou", "delta_ciou", "delta_mta"]
    for metric in metrics:
        plt.figure()
        plt.hist(delta_df[metric], bins=30, alpha=0.7, color='blue')
        plt.title(f'Histogram of {metric}')
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


def plot_delta(delta_df, metric):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(delta_df['instance_id'], delta_df[metric])
    plt.title(f'plot of {metric}')
    plt.xlabel('instance_id')
    plt.ylabel(metric)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    delta_df = compare_results(args.file1, args.file2)
    if delta_df is not None:
        print(delta_df)
        name1 = args.file1.split('/')[-1].replace('.xlsx', '')
        name2 = args.file2.split('/')[-1].replace('.xlsx', '')

        # save the differences to a new excel file
        # output_file = f"differences_{name1}-{name2}.xlsx"
        # delta_df.to_excel(output_file, index=False)
        # print(f"Differences saved to {output_file}")

        # plot histograms of the differences
        print(f"Plotting diagrams of the differences {name1} - {name2}")
        # plot_histograms(delta_df)
        plot_delta(delta_df, 'delta_polis')
        plot_delta(delta_df, 'delta_iou')
        plot_delta(delta_df, 'delta_ciou')
        plot_delta(delta_df, 'delta_mta')

    
