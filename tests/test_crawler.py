import os.path as osp

from movienet.tools import DoubanCrawler, IMDBCrawler, TMDBCrawler


class TestCrawler(object):

    @classmethod
    def setup_class(cls):
        cls.mid = 'tt0120338'
        cls.tmdb_id = '597'
        cls.douban_id = '1292722'

    def test_imdb_crawler(self):
        crawler = IMDBCrawler()
        info = crawler.parse_home_page(self.mid)
        credit_info = crawler.parse_credits_page(self.mid)
        synopsis_info = crawler.parse_synopsis(self.mid)
        info = {**info, **credit_info, **synopsis_info}
        assert info['title'] == 'Titanic (1997)'
        assert info['genres'] == ['Drama', 'Romance']
        assert info['country'] == 'USA'
        assert info['version'][0]['runtime'] == '194 min'
        assert info['director']['name'] == 'James Cameron'
        assert info['director']['id'] == 'nm0000116'
        assert info['cast'][0]['name'] == 'Leonardo DiCaprio'
        assert info['cast'][0]['id'] == 'nm0000138'
        assert len(info['synopsis'].split()) == 1444

    def test_douban_crawler(self):
        crawler = DoubanCrawler()
        douban_id = crawler.imdb2douban(self.mid)
        assert douban_id == self.douban_id

    def test_tmdb_crawler(self):
        if osp.isfile('apikey.txt'):
            crawler = TMDBCrawler('apikey.txt')
            tmdb_id = crawler.imdb2tmdb(self.mid)
            assert tmdb_id == self.tmdb_id
