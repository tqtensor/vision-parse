import os
from pathlib import Path
import nltk
from nltk.translate.bleu_score import sentence_bleu
from Levenshtein import distance
from markitdown import MarkItDown
from vision_parse import VisionParser
import statistics
from datetime import datetime

# Download required NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")


def calculate_bleu_score(reference_text: str, candidate_text: str) -> float:
    """Calculate BLEU score between reference and candidate text."""
    reference_tokens = nltk.word_tokenize(reference_text.lower())
    candidate_tokens = nltk.word_tokenize(candidate_text.lower())
    return sentence_bleu([reference_tokens], candidate_tokens)


def calculate_levenshtein_similarity(reference_text: str, candidate_text: str) -> float:
    """Calculate normalized Levenshtein similarity between two texts."""
    max_len = max(len(reference_text), len(candidate_text))
    if max_len == 0:
        return 1.0
    return 1 - (distance(reference_text, candidate_text) / max_len)


def benchmark_parser(
    parser_name: str,
    parser_func,
    pdf_path: Path,
    ground_truth: str,
    num_runs: int = 3,
) -> dict:
    """Benchmark a specific parser's performance."""
    accuracy_scores = []

    for i in range(num_runs):
        result = parser_func(pdf_path)

        # Extract text content from parser result
        if hasattr(result, "text_content"):
            parsed_text = result.text_content
        elif isinstance(result, list):
            parsed_text = "\n".join(result)
        else:
            parsed_text = str(result)

        # Calculate performance metrics
        bleu_score = calculate_bleu_score(ground_truth, parsed_text)
        levenshtein_score = calculate_levenshtein_similarity(ground_truth, parsed_text)
        accuracy = (bleu_score + levenshtein_score) / 2

        accuracy_scores.append(accuracy)

    # Calculate average metrics
    avg_accuracy = statistics.mean(accuracy_scores)

    return {
        "parser": parser_name,
        "avg_accuracy": avg_accuracy,
        "num_runs": num_runs,
        "individual_runs": {
            "accuracy_scores": accuracy_scores,
        },
    }


def save_benchmark_results(results: list, output_file: Path):
    """Save benchmark results to a markdown file."""
    with open(output_file, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Write summary table
        f.write("## Summary\n\n")
        f.write("| Parser | Accuracy |\n")
        f.write("|--------|----------|\n")
        for result in results:
            f.write(f"| {result['parser']} | {result['avg_accuracy']:.2f} |\n")

        # Write detailed results
        f.write("\n## Detailed Results\n\n")
        for result in results:
            f.write(f"### {result['parser']}\n\n")
            f.write(f"- Number of runs: {result['num_runs']}\n")
            accuracy_str = ", ".join(
                [f"{x:.2f}" for x in result["individual_runs"]["accuracy_scores"]]
            )
            f.write(f"- Individual accuracy scores: {accuracy_str}\n\n")


def main():
    # Configure input/output paths
    pdf_path = Path("quantum_computing.pdf")
    benchmark_results_path = Path("benchmark_results.md")

    with open(Path("ground_truth.md"), "r") as f:
        ground_truth = f.read()

    # Initialize parsers
    vision_parser = VisionParser(
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        top_p=0.5,
        image_mode=None,
        detailed_extraction=False,
        enable_concurrency=True,
    )
    markitdown_parser = MarkItDown()

    # Define parser functions
    def vision_parse_func(path):
        return vision_parser.convert_pdf(path)

    def markitdown_func(path):
        return markitdown_parser.convert(str(path))

    # Run benchmarks
    results = []
    results.append(
        benchmark_parser("Vision Parse", vision_parse_func, pdf_path, ground_truth)
    )
    results.append(
        benchmark_parser("Markitdown", markitdown_func, pdf_path, ground_truth)
    )

    # Save results to file
    save_benchmark_results(results, benchmark_results_path)

    # Print results to console
    print("\nBenchmark Results:")
    print("-" * 80)
    print(f"{'Parser':<15} {'Accuracy':<15}")
    print("-" * 80)

    for result in results:
        print(f"{result['parser']:<15} " f"{result['avg_accuracy']:<15.2f} ")

    print(f"\nDetailed results saved to: {benchmark_results_path}")


if __name__ == "__main__":
    main()
