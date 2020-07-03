import random
import time
import urllib

import requests
from bs4 import BeautifulSoup
from fake_useragent import FakeUserAgentError, UserAgent


class DoubanCrawler(object):

    def __init__(self):
        self.url_prefix = 'https://movie.douban.com/'
        try:
            ua = UserAgent()
            self.header = {'User-Agent': str(ua.chrome)}
        except FakeUserAgentError:
            self.header = {'User-Agent': ''}

    def parse_runtime(self, douban_id):
        """get runtime of a movie from douban homepage.

        Args:
            douban_id (str): Douban ID

        Returns:
            list: runtimes and descritopns of each version
        """
        try:
            response = requests.get(
                '{}/subject/{}'.format(self.url_prefix, douban_id),
                headers=self.header)
            content = response.text
        except Exception as e:
            print(e)
        ret = []
        page_soup = BeautifulSoup(content, 'lxml')
        try:
            span = page_soup.find_all('span', {'property': 'v:runtime'})
            assert len(span) == 1
            runtime_str = span[0].get_text().strip()
            other_runtimes = span[0].next_sibling
            if str(other_runtimes).find('br') < 0:
                other_runtimes = other_runtimes.strip()
                runtime_str += other_runtimes
            runtime_str = runtime_str.replace('分钟', '').strip()
            runtime_list = runtime_str.split('/')
            for info in runtime_list:
                if info.find(':') >= 0:
                    descrption, runtime = info.split(':')
                elif info.find('(') >= 0:
                    runtime, descrption = info.split('(')
                    descrption = descrption[:-1]
                else:
                    descrption = ''
                    runtime = info
                ret.append({
                    'runtime': '{} min'.format(int(runtime.strip())),
                    'description': descrption.strip()
                })
        except Exception as e:
            print('{} {}'.format(douban_id, e))
        return ret

    def douban2imdb(self, douban_id):
        """Convert a Douban ID to IMDB ID.

        Args:
            douban_id (str): Douban ID

        Returns:
            str: IMDB ID
        """
        url = 'https://movie.douban.com/subject/{}'.format(douban_id)
        response = requests.get(url, headers=self.header)
        time.sleep(random.randint(0, 3))
        html = response.content.decode('utf-8')
        soup = BeautifulSoup(html, 'lxml')
        all_info = soup.find('div', id='info')
        if all_info is None:
            return None
        for each_info in all_info.find_all('a'):
            if each_info.attrs.get('href').find('imdb') > 0:
                imdbid_pos = str(each_info).rfind('</a>')
                imdbid = str(each_info)[imdbid_pos - 9:imdbid_pos]
                print(imdbid)
        return imdbid

    def imdb2douban(self, mid):
        """Convert an IMDB ID to Douban ID.

        Args:
            mid (str): IMDB ID

        Returns:
            str: Douban ID
        """
        douban_id = None
        try:
            word = mid + ' 电影 豆瓣'
            url = 'http://www.baidu.com.cn/s?wd=' + urllib.parse.quote(
                word) + '&pn=0'
            htmlpage = urllib.request.urlopen(url).read()
            soup = BeautifulSoup(htmlpage, 'lxml')
            tagh3 = soup.find_all('h3')
            for h3 in tagh3:
                href = h3.find('a').get('href')
                baidu_url = requests.get(
                    url=href, allow_redirects=False, headers=self.header)
                real_url = baidu_url.headers['Location']
                request_title = 'https://movie.douban.com/subject/'
                if real_url.startswith(request_title):
                    ret = real_url.split(request_title)[1].split('/')[0]
                    print(ret)
                    if self.douban2imdb(ret) == mid:
                        douban_id = ret
                        break
        except Exception as e:
            print('{} {}'.format(mid, e))
        return douban_id
