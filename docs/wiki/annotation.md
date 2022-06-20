# Annotation in MovieNet

## Person

Person plays an important role in human-centric videos like movies.
Thus to detect and identify persons is a foundational work toward movie understanding.

We select 758K key frames from more than 300 movies and manually annotate the
bounding box of each person in each image, leading to **1.3M person bounding boxes**.

To draw a bounding box of a person is simple while to annotate the identities is much more complicated. Totally 573 movies are selected for identity annotation. For the other movies without bounding box annotation, we use a state-of-the-art person detector (see [movie-toolbox](/movie-toolbox/tools)) to detect them and manually remove the false positive results. And we only select 1 of the 3 key frames for identity annotation, by the observation that it is easy to track one person within a shot by bounding box overlap.
To make the cost aordable, we only keep the top 10 cast in credits order according to IMDb, which can cover the
main characters for most movies.
Persons who do not belong to credited cast would be labeled as "others".

In total, we obtained **763K instances** of **3,087 credited** cast and 364K "others".

## Scene Boundary

In MovieNet, we manually annotate the scene boundaries of more than 318 movies to support the researches on scene segmentation. Each boundary is annotated by three dierent annotators and only the boundaries achieved
high consistency would be kept, resulting in **42K scenes**.

## Place/Action Tag

We split each movie into segments according to scene boundaries and manually annotated tags of place and action for over each segment.

For place annotation, each segment is annotated with multiple place tags, e.g., fdeck, cabing.

While for action annotation, we first detect sub-clips that contain people and actions, then we assign multiple action tags
to each sub-clip.
We have made the following eorts to keep tags diverse and informative:
(1) We encourage the annotators to create new tags and
(2) Tags that convey little information for story understanding, e.g., stand and talk, are excluded.
Finally, we merge the tags and ltered out 80 action classes and 90 place classes with a minimum frequency of 25 as the final annotations.

In total, there are 13.7K segments with **19.6K scene tags** and **41.3K action clips** with **45K action tags**.

## Description Alignment

In MovieNet, we choose synopses as the story descriptions.

The associations between the movie segments and the synopsis paragraphs are manually annotated by three different annotators with a coarse-to-fine procedure.

Finally, we obtained **4,208 highly consistent paragraph-segment pairs** from 297 movies.

## Cinematic Style

Cinematic styles is an important aspect of comprehensive movie understanding,
which influences how the story is telling how the story is telling in a movie.

In MovieNet, we choose two kinds of cinematic tags for study now, namely view scale and camera movement.

Specically, the view scale include five categories, i.e. **long shot**, **full shot**, **medium shot**, **close-up shot** and **extreme close-up shot**.

While the camera movement is divided into four classes,
i.e. **static shot**, **pans and tilts shot**, **zoom in** and **zoom out**. The original denitions of these categories come from the book _Understanding movies_ (Louis D Giannetti and Jim Leach, 1991) and we simplify them for research
convenience.

MovieNet contains **47K shots** from around **8K movies and trailers**, each with one tag of view scale and one tag of camera movement.

## An Example of Annotation

Here we give the annotation of the movie _Moneyball_ as an example.

```json
{
  "imdb_id": "tt1210166",
  "cast": [
    {
      "id": "tt1210166_000001",
      "frame_idx": null,
      "resolution": [
        1280,
        694
      ],
      "shot_idx": 1,
      "img_idx": 0,
      "body": {
        "type": "detected",
        "bbox": [
          22,
          27,
          1148,
          675
        ]
      },
      "pid": "others",
      "possible_pids": [
        "others"
      ]
    },
    ...
  ],
  "scene": [
    {
      "id": "tt1210166_0000",
      "shot": [
        0,
        1
      ],
      "frame": [
        0,
        841
      ],
      "place_tag": null,
      "action_tag": null
    },
    ...
  ],
  "story": [
    {
      "id": "tt1210166_0000",
      "shot": [
        60,
        424
      ],
      "frame": [
        6257,
        44851
      ],
      "duration": [
        260.97997833333335,
        1870.6211273333333
      ],
      "consistency": 0.963081028938084,
      "description": "Oakland Athletics general manager Billy Beane is upset by his team's loss to the New York Yankees in the 2001 postseason ...",
      "subtitle": [
        {
          "shot": 60,
          "duration": [
            260.26,
            262.51225
          ],
          "sentences": [
            "You gotta give the Yankees--"
          ]
        },
        ...
      ]
    },
    ...
  ],
  "cinematic_style": {
    "movie": [
      {
        "shot": 1,
        "scale": "closeup",
        "movement": "static"
      },
      {
        "shot": 2,
        "scale": "full",
        "movement": "static"
      },
      {
        "shot": 3,
        "scale": "closeup",
        "movement": "moving"
      },
      ...
    ],
    "trailer": null
  }
}
```
