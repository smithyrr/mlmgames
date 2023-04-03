import requests
from bs4 import BeautifulSoup
import json

base_url = "https://community.bistudio.com"
url = "https://community.bistudio.com/wiki/Category:Scripting_Commands_by_Functionality"
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")
categories = soup.find_all("div", class_="CategoryTreeItem")

commands_data = []

for category in categories:
    category_url = base_url + category.find("a")["href"]
    category_response = requests.get(category_url)
    category_soup = BeautifulSoup(category_response.text, "html.parser")

    commands_list = category_soup.find_all("div", class_="mw-category-group")

    for group in commands_list:
        commands = group.find_all("a")
        for command in commands:
            command_url = base_url + command["href"]
            command_response = requests.get(command_url)
            command_soup = BeautifulSoup(command_response.text, "html.parser")

            command_name = command.text.strip()

            description = command_soup.find("div", class_="description")
            if not description:
                description = command_soup.find("p", class_="description")
            if not description:
                description = command_soup.find("div", class_="description_content")
            if not description:
                description = command_soup.find("p", class_="plainlinks")
            if not description:
                description = command_soup.find("div", class_="quote")
            command_description = description.get_text(strip=True) if description else "No description available."

            example = command_soup.find("div", class_="sqfhighlighter-scroller")
            command_example = example.get_text(strip=True) if example else "No example available."

            commands_data.append({
                "name": command_name,
                "description": command_description,
                "example": command_example
            })

            print(f"Command: {command_name}\nDescription: {command_description}\nExample: {command_example}\n\n")

with open("arma3_commands_by_functionality.json", "w") as file:
    json.dump(commands_data, file, indent=4)
