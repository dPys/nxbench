import asyncio
import logging
import os
import random

import pandas as pd

from nxbench.data.repository import NetworkRepository, NetworkStats

logger = logging.getLogger("nxbench")


async def main(
    test_mode: bool = False,
    test_limit: int = 5,
    db_file: str = "../network_directory.csv",
    max_concurrent_tasks: int = 3,
):
    if os.path.exists(db_file):
        existing_df = pd.read_csv(db_file)
        processed_networks = set(existing_df["name"])
        logger.info(f"Found {len(processed_networks)} already processed networks.")
    else:
        processed_networks = set()
        existing_df = None
        logger.info("No existing network_directory.csv found. Processing all networks.")

    async with NetworkRepository() as repo:
        networks_data = []
        networks_by_category = repo.networks_by_category

        if test_mode:
            max_networks = test_limit
            logger.info(
                f"Running in test mode: Processing up to {max_networks} networks."
            )
            all_networks = [
                (category, name)
                for category, network_names in networks_by_category.items()
                for name in network_names
                if name not in processed_networks
            ]
            limited_networks = all_networks[:max_networks]
        else:
            limited_networks = [
                (category, name)
                for category, network_names in networks_by_category.items()
                for name in network_names
                if name not in processed_networks
            ]

        logger.info(f"Total networks to process: {len(limited_networks)}")

        async def process_with_backoff(semaphore, task, max_retries=3):
            """Execute task with exponential backoff and circuit breaker."""
            for attempt in range(max_retries):
                try:
                    async with semaphore:
                        return await asyncio.wait_for(
                            task, timeout=120
                        )  # 2 min timeout
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = min(2**attempt + random.uniform(0, 1), 60)
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e!s}"
                    )
                    await asyncio.sleep(delay)

        async def process_network(name, category):
            try:
                metadata = await repo.get_network_metadata(name, category)
                url = metadata.download_url
                if not url:
                    logger.warning(
                        f"No download URL found for network '{name}'. Skipping."
                    )
                    return

                data = metadata.__dict__.copy()
                if isinstance(data.get("network_statistics"), NetworkStats):
                    data["network_statistics"] = data["network_statistics"].__dict__
                networks_data.append(data)
                logger.info(f"Processed network '{name}' in category '{category}'")
            except Exception as e:
                logger.exception(
                    f"Error processing network '{name}' in category '{category}': {e}"
                )

        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        tasks = [
            process_with_backoff(semaphore, process_network(name, category))
            for category, name in limited_networks
        ]

        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            await asyncio.gather(*batch, return_exceptions=True)

        df = pd.DataFrame(networks_data)

        if "network_statistics" in df.columns and isinstance(
            df["network_statistics"].iloc[0], dict
        ):
            stats_df = pd.json_normalize(df["network_statistics"])
            df = pd.concat([df.drop(columns=["network_statistics"]), stats_df], axis=1)

        if existing_df is not None:
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["name"], keep="last")
        else:
            combined_df = df

        combined_df.to_csv(db_file, index=False)
        logger.info(f"All metadata and download URLs have been saved to '{db_file}'")

        return combined_df


if __name__ == "__main__":
    df = asyncio.run(main(test_mode=False, test_limit=5, max_concurrent_tasks=3))
