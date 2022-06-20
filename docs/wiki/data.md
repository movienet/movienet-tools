# Data in MovieNet

## ID

All movies in MovieNet is indexed by its IMDb ID, i.e. the IMDb ID of "Titanic" is tt0120338.

[IMDb](https://www.imdb.com) is the world's most popular and authoritative source for movie and IMDb ID is the unique ID for each movie in this website.

Besides IMDb ID, we also provide [TMDb](https://www.themoviedb.org/) ID and [Douban](https://movie.douban.com/) ID for some of the movies. These two website also contains some valuable data for movie analysis.

## Movie

MovieNet now contains 1,100 movies. All the movies are resized to 720P, i.e. the height of video is 720.

And some of the movies would have black boarder in order to keep the ratio as 16:9. We remove these black boarders.

Considering the legal issue, we can only release the key frames but not the orginal movies.
To be specific, we first detect the shot boundaries and then extract three frames of each shot as keyframes.
Considering that the frames within each shot is quite similar, we think the keyframes can already support most of the researches in movie understanding.

Besides the keyframes, the audio waves are aslo released to support the research topics on multi-modalities.
To avoid copyright issue, we process the audio wave with stft in 16K Hz sampling rate and 512 window length.

## Trailer

Trailer is a commercial advertisement for a feature film, which usually contains some attractive moments in the movie.

Totally, we collect 33K unique trailers, from which key frames in each shot as well as audio waves are extracted like the movies.

## Subtitle

The subtitles in MovieNet are obtained in two ways.

Some of them are extracted from the embedded subtitle stream in the movies.

For movies without original English subtitle, we crawl the subtitles from [YIFY](https://www.yifysubtitles.com/).

All the subtitles are manually checked to ensure that they are aligned to the movies.

## Script

Script, where the movement, actions, expression and dialogs of the characters are narrated, is a valuable textual source for research topics of movie-language association.

We collect around 2K scripts from [IMSDb](https://www.imsdb.com/) and [Daily Script](ttps://www.dailyscript.com/).
We obtain 982 scripts by matching the dialog with subtitles and ltering noisy ones.

**\*TODO: More Details**

## Synopsis

A synopsis is a description of the story in a movie written by audiences.

We collect 31K synopses from IMDb. We only keep those contain more than 50 sentences by the evidence that the long synopses are usually with high quality, leading to 11K valid synopses left in MovieNet.

**\*TODO: More Details**

## Meta Data

MovieNet contains meta data of 375K movies from IMDb and
TMDb including title, release date, country, genres, rating, runtime, director,
cast, storyline, etc.

Genres is the most important attribute of a movie.
There are total 805K genres tags from 28 unique genres in MovieNet.

For the cast, we get both their names, IMDb IDs and the character names in
the movie.

For a movie with more than one versions, e.g. normal version,
director's cut, we also get the runtime and description of each version to help
researchers align the annotations.

Here we provide the meta data of _Titanic_ in MovieNet as an example.

```json
{
  "imdb_id": "tt0120338",
  "tmdb_id": "597",
  "douban_id": "1292722",
  "title": "Titanic (1997)",
  "genres": [
    "Drama",
    "Romance"
  ],
  "country": "USA",
  "version": [
    {
      "runtime": "194 min",
      "description": ""
    }
  ],
  "imdb_rating": 7.7,
  "director": [
    {
      "id": "nm0000116",
      "name": "James Cameron"
    }
  ],
  "writer": [
    {
      "id": "nm0000116",
      "name": "James Cameron",
      "description": "written by"
    }
  ],
  "cast": [
    {
      "id": "nm0000138",
      "name": "Leonardo DiCaprio",
      "character": "Jack Dawson"
    },
    {
      "id": "nm0000701",
      "name": "Kate Winslet",
      "character": "Rose Dewitt Bukater"
    },
    ...
  ],
  "overview": "84 years later, a 101-year-old woman named Rose DeWitt Bukater tells the story ..."
}
```
