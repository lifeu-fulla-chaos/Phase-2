import matplotlib.pyplot as plt


def read_file(filepath):
    """Read numerical data from a file."""
    with open(filepath, "r") as file:
        data = [list(map(float, line[1:-2].strip().split())) for line in file if line.strip()]
    return data


def plot_data(master_data, master_copy_data):
    """Plot data from master.txt and master-copy.txt."""
    master_data = list(zip(*master_data))  # Transpose to separate columns
    master_copy_data = list(zip(*master_copy_data))  # Transpose to separate columns

    plt.figure(figsize=(12, 6))

    for i, (master_col, master_copy_col) in enumerate(
        zip(master_data, master_copy_data)
    ):
        plt.subplot(len(master_data), 1, i + 1)
        plt.plot(master_col, label="Master", color="blue")
        plt.plot(master_copy_col, label="Master Copy", color="orange")
        plt.title(f"State {i + 1}")
        plt.xlabel("Time Step")
        plt.ylabel(f"Value {i + 1}")
        plt.legend()

    plt.tight_layout()
    plt.savefig("master-vs-master-copy.png")


def main():
    master_data = read_file("master.txt")
    master_copy_data = read_file("master-copy.txt")
    plot_data(master_data, master_copy_data)


main()
