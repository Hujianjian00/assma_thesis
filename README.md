# assma_thesis
I first experimented with the deepseek-r1-7b model on the Enron email dataset, but the results were weaker than expected. To improve performance, I switched to the deepseek-llm-7b-base model and expanded testing to include both the Enron emails and the AG News dataset.
# Challenges faced:
- **Hardware Limitations**: Despite having a reasonably equipped system **(8GB GPU, 32GB RAM, Intel Core Ultra 7 155H, 1.86TB SSD)**, the computational demands are overwhelming. Fine-tuning and executing attacks on even a 1K-sample dataset took over half a week. With larger datasets and more complex models, my current setup is insufficient to meet deadlines.
- **Technical Hurdles**:  As I am new to fine-tuning open-source LLMs, I encountered several coding obstacles, most of which I resolved through research and experimentation. However, I am currently stuck on an issue where the script fails to properly recognize the dataset, despite the environment detecting it.
# Project Structure:
-ag_news folder: Contains train/test parquet files for the AG News dataset.

-emails.csv: Holds the Enron email dataset for testing.

-attack folder: Member interference attack script.

-cache folder: Stores cached model weights from prior email dataset training.

-configs folder: Configuration scripts for global arguments.

-data folder: Data preprocessing scripts.

-ft_llms folder: Fine-tuning scripts.

# Requests:
- Could you help me troubleshoot the dataset recognition issue in my script so it will run with no error

- I kindly request access to a more powerful server or cloud-based GPU resources to accelerate fine-tuning and experimentation. The deadline is approaching quickly, and I want to ensure the project meets its goals efficiently.
