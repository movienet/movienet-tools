import requests
from bs4 import BeautifulSoup
import json
import os
import os.path as osp
from termcolor import colored

class IMDBCrawler(object):
    def __init__(self):
        self.url_prefix = 'https://www.imdb.com/title'

    def parse_home_page(self, mid):
        """get infomation of a movie from its IMDB homepage,
           including title, storyline, genres, country and version
        
        Args:
            mid (str): IMDB ID
        
        Returns:
            dict: information dict
        """
        try:
            response = requests.get('{}/{}'.format(self.url_prefix, mid))
            content = response.text
        except Exception as e:
            print(e)
        page_soup = BeautifulSoup(content, 'lxml')
        info = {
            'imdb_id': mid,
            'title': None,
            'genres': None,
            'country': None,
            'version': [],
            'storyline': None
            }
        # get title
        try:
            title_meta = page_soup.find('meta', property='og:title')
            title = title_meta['content']
            info['title'] = title.split('-')[0].strip()
        except Exception as e:
            print('Cannot find title. {}: {}'.format(mid, e))
        # get storyline and genres
        try:
            titleStoryLine = page_soup.find('div', {'id':'titleStoryLine'})
            divs = titleStoryLine.find_all('div', class_='canwrap')
            for div in divs:
                p = div.find('p')
                if p is not None:
                    story = p.find('span').get_text()
                    info['storyline'] = story.strip()
                else:
                    title = div.find('h4')
                    if title is not None:
                        title = title.string
                        if title.find('Genres') >= 0:
                            # print(title)
                            genres = []
                            genres_as = div.find_all('a')
                            for genres_a in genres_as:
                                genres.append(genres_a.string.strip())
                            info['genres'] = genres
        except Exception as e:
            print(colored('Cannot find storyline or genres. {}: {}'.format(mid, e), 'red'))
        # get country and runtime
        try:
            titleStoryLine = page_soup.find('div', {'id':'titleDetails'})
            divs = titleStoryLine.find_all('div', class_='txt-block')
            for div in divs:
                title = div.find('h4')
                if title is not None:
                    title = title.string
                    if title.find('Country') >= 0:
                        country_a = div.find('a')
                        info['country'] = country_a.string.strip()
                    if title.find('Runtime') >= 0:
                        text = div.get_text().strip().replace('Runtime:', '')
                        runtimes = text.split('|')
                        for runtime in runtimes:
                            minutes = runtime.split('min')[0].strip()
                            description = runtime.split('min')[1].strip()
                            description = description.replace('(', '')
                            description = description.replace(')', '')
                            info['version'].append({
                                'runtime': minutes + ' min',
                                'description': description
                            })
        except Exception as e:
            print(colored('Cannot find country or runtime. {}: {}'.format(mid, e), 'red'))
        return info
    
    def parse_credits_page(self, mid):
        """get infomation of a movie from its IMDB credits page,
           including director and cast
        
        Args:
            mid (str): IMDB ID
        
        Returns:
            dict: information dict
        """
        info = {
            'imdb_id': mid,
            'cast': None,
            'director': None,
            }
        try:
            response = requests.get('{}/{}/{}'.format(self.url_prefix, mid, 'fullcredits'))
            content = response.text
            page_soup = BeautifulSoup(content, 'lxml')
            credit_div = page_soup.find('div', {'id': 'fullcredits_content'})
            # find director
            d_table = credit_div.find('table', class_='simpleCreditsTable')
            td = d_table.find('td', class_='name')
            tda = td.find('a')
            pid = tda['href'].split('/')[2]
            name = tda.string.strip()
            info['director'] = {'id': pid, 'name': name}
            # find cast
            cast_table = credit_div.find('table', class_='cast_list')
            if cast_table is not None:
                trs = cast_table.find_all('tr')
                cast = []
                for tr in trs:
                    td = tr.find('td', class_='primary_photo')
                    if td is None:
                        continue
                    tda = td.find('a')
                    pid = tda['href'].split('/')[2].strip()
                    name = tda.find('img')['title'].strip()
                    td = tr.find('td', class_='character')
                    character = td.get_text().strip()
                    character = character.replace('\n','')
                    if character.find('uncredited') >= 0:
                        continue
                    cast.append({'id': pid, 'name': name, 'character': character})
                info['cast'] = cast
        except Exception as e:
            print(colored('Cannot find credits. {}: {}'.format(mid, e), 'red'))
        return info
    
    def parse_synopsis(self, mid):
        """get synopsis of a movie
        
        Args:
            mid (str): IMDB ID
        
        Returns:
            dict: synopsis
        """
        info = {
            'synopsis': None
        }
        try:
            response = requests.get('{}/{}/plotsummary'.format(self.url_prefix, mid))
            content = response.text
            page_soup = BeautifulSoup(content, 'lxml')
            ul = page_soup.find('ul', {'id': 'plot-synopsis-content'})
            synopsis = ul.get_text().strip()
            if synopsis.find('Edit page') >= 0:
                synopsis = None
            info['synopsis'] = synopsis
        except Exception as e:
            print(colored('{} {}'.format(mid, e), 'red'))
        return info
    
    # def parse_and_save(self, mid, save_dir='.'):
    #     try:
    #         info = self.parse_home_page(mid)
    #         credit_info = self.parse_credits_page(mid)
    #         synopsis_info = self.parse_synopsis(mid)
    #         info = {**info, **credit_info, **synopsis_info}
    #         with open(osp.join(save_dir, '{}.json'.format(mid)), 'w') as f:
    #             f.write(json.dumps(info, indent=2))
    #     except Exception as e:
    #         print('{} {}'.format(mid, e))
