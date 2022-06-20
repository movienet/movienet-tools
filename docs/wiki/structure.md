---
order: 2
---

# Structure of a Movie

## Temporal Sturcture

From the pespective of temporal sturcture, a movie would has a hierachical structure:

- **frame -> shot -> (thread ->) scene -> movie**

**Shot** is a series of frames that runs for an uninterrupted period of time.
It is also the minimal visual unit of a movie.
A movie would usually contains hundreds of shots.

**Scene** is a sequence of continued shots that are semantically related.
Usually a scene would tell about one event in the movie.
A movie would contains tens of scenes.

**Thread** shows the pattern of the shot arrangement in a scene.
But note that not all scenes would contain threads.

Take a typical dialog scene as an example. Suppose there are two persons
A and B in the dialog scene, they would be alternately shown, the pattern
of which can be represented as ABABAB...". So there are two threads in
this dialog scene, namely A and B. To capture the hierarchical structure of
a movie is important for movie understanding.

## Content Sturcture

From the perspective of content structure, movie is an **art of storytelling**.

In terms of story, a movie contains three key elements, namely **person**, **place** and **action**.

In terms of art, each shot in a movie would contain specific **cinematic style**, e.g. the view scale and the camera movement.

&nbsp;
&nbsp;

All the data and annotations in MovieNet are built based on these two pespectives.
