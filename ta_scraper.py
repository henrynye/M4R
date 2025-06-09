import requests
from bs4 import BeautifulSoup
import os
import re
import csv
from tqdm import tqdm
import time


base_url = "https://www.tennisabstract.com/cgi-bin/player-classic.cgi?p="
output_dir = "tennis_abstract"
gmp_name = 'MarianoNavone'
# gmp_name = 'GiovanniMpetshiPerricard'
formatted_names = 'formmatted_names.csv'
formatted_names_path = f'{output_dir}/{formatted_names}'


def download_webpage(url, output_filename):
    """
    Downloads the HTML content of a webpage and saves it to a file.

    Args:
      url: The URL of the webpage to download.
      output_filename: The name of the file to save the HTML content to.
    """
    # Send an HTTP GET request to the URL
    response = requests.get(url)
    time.sleep(3)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        try:
            # Get the HTML content from the response
            html_content = response.text

            # Create the output directory if it doesn't exist
            output_dir = os.path.dirname(output_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the HTML content to the specified file
            with open(output_filename, "w", encoding="utf-8") as file:
                file.write(html_content)

            # Optionally: Parse the HTML content using BeautifulSoup (if needed)
            soup = BeautifulSoup(html_content, "html.parser")
            script_tags = soup.find_all("script")
            ochoices = None
            for script in script_tags:
                if script.string:  # Ensure the script tag contains JavaScript code
                    match = re.search(r'var\s+ochoices\s*=\s*(\[.*?\]);', script.string, re.DOTALL)
                    if match:
                        # Extract the JavaScript array or object
                        ochoices_str = match.group(1)
                        # Convert the JavaScript array/object to a Python object
                        ochoices = eval(ochoices_str)
                        formatted_names = set([n.replace(" ", "") for n in ochoices])
                        with open(formatted_names_path, 'r+', newline='') as file:
                            reader = csv.reader(file)
                            og_names = set([row[0].strip().rstrip(',') for row in reader])
                            og_names.update(formatted_names)

                            file.seek(0)
                            file.truncate()

                            writer = csv.writer(file)
                            writer.writerows([[n] for n in og_names])
                        break
        except Exception as e:
            breakpoint()

# Example usage:
# formatted_names = [n.replace(" ", "") for n in gmp_player_names]
flag = True
while flag:
    flag = False
    existing_files = [f'{output_dir}/{n}' for n in os.listdir(output_dir)]
    with open(formatted_names_path, 'r', newline='') as file:
        reader = csv.reader(file)
        names = [row[0].strip().rstrip(',') for row in reader]
    for name in tqdm(names):
        output_filename = f"{output_dir}/{name}.html"
        if output_filename not in existing_files:
            download_webpage(f"{base_url}{name}", output_filename)
            flag = True
# For a nested path: output_filename = "tennis_data/players/tennis_player_page.html"

# download_webpage(url, output_filename)