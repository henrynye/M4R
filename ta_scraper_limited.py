import requests
from bs4 import BeautifulSoup
import os
import re
import csv
from tqdm import tqdm
import time
import json


base_url = "https://www.tennisabstract.com/cgi-bin/player-classic.cgi?p="
output_dir = "tennis_abstract"
gmp_name = 'MarianoNavone'
# gmp_name = 'GiovanniMpetshiPerricard'
formatted_names_limited = 'formmatted_names_limited.csv'
formatted_names_path = f'{output_dir}/{formatted_names_limited}'
global count
global count2
count, count2 = 0, 0 

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

def get_ochoices_locally(name):
    file_path = f'{output_dir}/{name}.html'
    if not os.path.exists(file_path):
        return set()
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    script_tags = soup.find_all("script")
    formatted_names = None
    for script in script_tags:
        if script.string:  # Ensure the script tag contains JavaScript code
            match = re.search(r'var\s+ochoices\s*=\s*(\[.*?\]);', script.string, re.DOTALL)
            if match:
                # Extract the JavaScript array or object
                ochoices_str = match.group(1)
                # Convert the JavaScript array/object to a Python object
                ochoices = eval(ochoices_str)
                formatted_names = set([n.replace(" ", "") for n in ochoices])
    if formatted_names is None:
        return set()
    return formatted_names

# Example usage:
# formatted_names = [n.replace(" ", "") for n in gmp_player_names]
with open('api_p_keys_to_ta_names.json', 'r') as file:
    players = json.load(file)

with open('formatted_names_deg_3.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    formatted_names_deg_3 = [r[0].strip().rstrip(',') for r in reader]


# ta_names = [v for k,v in players.items()]
# formatted_names_deg_2 = set()
# for name in ta_names:
#     formatted_names_deg_2.update(get_ochoices_locally(name))

# formatted_names_deg_3 = set()
# for name in tqdm(formatted_names_deg_2):
#     formatted_names_deg_3.update(get_ochoices_locally(name))

# print('Finished getting 2nd degree players')
existing_files = [f'{output_dir}/{n}' for n in os.listdir(output_dir)]
for name in tqdm(formatted_names_deg_3):
    output_filename = f"{output_dir}/{name}.html"
    if output_filename not in existing_files:
        download_webpage(f"{base_url}{name}", output_filename)


# For a nested path: output_filename = "tennis_data/players/tennis_player_page.html"

# download_webpage(url, output_filename)