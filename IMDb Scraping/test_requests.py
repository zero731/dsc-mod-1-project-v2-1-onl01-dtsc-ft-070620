"""Code written by Sam Stoltenberg - github.com/skelouse"""
import re
import json
import requests
from bs4 import BeautifulSoup
file_name = 'name.txt'
data = {}

def extract_numbers(string):
    try:
        return int("".join(re.findall(r'\b\d+\b', string)))
    except ValueError:
        return None


def find(soup, q_num):
    budget = None
    gross = None
    ww_gross = None
    rating = None
    # Find the rating
    for div in soup.findAll('div', class_='subtext'):
        print(div.text.split('\n')[1])
        rating = (div.text.split('\n')[1].replace(' ', '').replace('\n', ''))
        print(rating)
    # Find the budget, gross, ww_gross in page
    for h4 in soup.findAll('h4'):
        if h4.text.startswith('Budget'):
            text = h4.parent.text
            text = text.split(' ')[0]
            budget = extract_numbers(text)
            
        elif h4.text.startswith('Gross USA'):
            text = h4.parent.text
            text = text.split(' ')[2]
            gross = extract_numbers(text)
            
        elif h4.text.startswith('Cumulative Worldwide'):
            text = h4.parent.text
            text = text.split(' ')[3]
            ww_gross = extract_numbers(text)
    if budget or gross or ww_gross or rating:
        new_data = {
            q_num:{
                'budget': budget,
                'gross': gross,
                'ww_gross': ww_gross,
                'rating': rating
                }
            }
        data.update(new_data)

url = "https://www.imdb.com/title/"
def get_soup(q_num):
    req = requests.get(str(url+q_num))
    return BeautifulSoup(req.content.decode(), 'html.parser')

def save(data):
    with open('data.json', 'w') as f:
        json.dump(data, f)

with open(file_name, 'r') as f:
    q_list = f.read().split(',')

x = 0
for num in q_list:
    x += 1
    if x == 100:
        x = 0
        save(data)
    print('analysing', num)
    find(get_soup(num), num)

save(data)

