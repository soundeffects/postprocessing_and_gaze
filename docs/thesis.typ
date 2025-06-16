// Preliminary counter
#let preliminary = counter("preliminary")

// Initial styling
#set page(
  header: context {
    let current_page = counter(page).get()
    let current_preliminary = counter("preliminary").get()

    // First, check page if there is a heading
    let heading_query = query(selector(heading))
    let heading_present = heading_query.any(item => {
      let item_page = counter(page).at(item.location())
      let item_preliminary = counter("preliminary").at(item.location())
      item_page == current_page and item_preliminary == current_preliminary
    })

    // Second, check page if there is a no-header tag
    let no_header_query = query(<no-header>)
    let no_header_present = no_header_query.any(item => {
      let item_page = counter(page).at(item.location())
      let item_preliminary = counter("preliminary").at(item.location())
      item_page == current_page and item_preliminary == current_preliminary
    })

    if not heading_present and not no_header_present [
      #h(1fr) #counter(page).display()
    ]
  },
  header-ascent: 40%,
  number-align: right,
  margin: (
    x: 1.5in,
    y: 1in,
  ),
  numbering: "i",
  footer: "",
)

#set text(
  size: 12pt,
  font: "Times New Roman",
)

#set par(
  spacing: 1em,
  leading: 1em,
)

// Metadata
#let author = "James Youngblood"
#let year = "2025"
#let month = "August"
#let degree = "Master of Science"
#let department = "School of Computing"
#let title = "GAZE FIXATION PREDICTION FOR COMPOSABLE IMAGE TRANSFORMATIONS"
#let abstract = [  
  We contribute simple heuristic functions for estimating how much gaze fixation prediction models adhere to real gaze distributions under an arbitrary composition of image transformations. Notably, computing these heuristics does not require the collection of real gaze distributions. We analyze the effectiveness of the heuristics by computing their correlation to the divergence of predicted gaze distributions from real gaze distributions on a set of image transformations.
]
#let committee_chair = "Rogelio Cardona-Rivera"
#let committee_second = "Paul Rosen"
#let committee_third = "Cem Yuksel"
#let department_chair = "Mary Hall"
#let graduate_dean = "Darryl P. Butt"

#show heading: item => [
  #set align(center)
  #set text(14pt, weight: "regular")
  #set par(leading: 1em)
  #pagebreak(weak: true)
  #v(1in)
  #item
  #v(1em)
]

// Title page
#counter(page).update(0)
#align(center)[
  #text([#title], size: 14pt) <no-header>
  #v(1fr)
  by \
  #author
  #v(1fr)
  A thesis submitted to the faculty of \
  The University of Utah \
  in partial fulfillment of the requirements for the degree of
  #v(1fr)
  #degree
  #v(1fr)
  #department \
  The University of Utah \
  #month #year
]
#pagebreak()

// Copyright page
#align(center)[
  #v(1fr)
  Copyright © <no-header> #author #year \
  All Rights Reserved 
  #v(1fr)
]
#pagebreak()

// Approval page
#align(center)[
  *The University of Utah Graduate School* <no-header>
  #v(3em)
  STATEMENT OF THESIS APPROVAL
  #v(3em)
]
The thesis of #emph([#author]) has been approved by the following supervisory committee members:
#v(1em)
#grid(
  columns: (auto, auto, 1fr),
  gutter: 1em,
  [#emph([#committee_chair]),],
  [Chair,],
  [_(Date Approved)_],
  [#emph([#committee_second]),],
  [Member,],
  [_(Date Approved)_],
  [#emph([#committee_third]),],
  [Member,],
  [_(Date Approved)_],
)
#v(1em)
by #emph([#department_chair]), the Chair of the #department, \
and by #emph([#graduate_dean]), the Dean of the Graduate School.
#v(1fr)
#pagebreak()

// Abstract page
= ABSTRACT
#h(2em)#abstract
#pagebreak()

// Table of Contents
#outline(
  title: [CONTENTS #v(2em)],
  indent: auto,
  depth: 2,
)
#pagebreak()

// List of Figures
= LIST OF FIGURES
#outline(
  title: none,
  target: figure.where(kind: image),
)
#pagebreak()

// Setup for main content
#set page(numbering: "1")
#set par(first-line-indent: (amount: 2em, all: true))
#set heading(numbering: "I.")
#counter(page).update(1)
#counter("preliminary").update(1)

= INTRODUCTION
Visual media producers study the process of viewing. Studios which produce movies, games, illustration, or other visual media have a professional interest in how a viewer will direct their gaze over the presented image. With better knowledge, they can produce pleasing image compositions (i.e. the "rule of thirds"), or estimate the viewer's situational awareness (i.e. noticing a figure in the dark). In order to exert maximum control over the viewing experience, producers must intuit the complexities of human visual behavior, or gather a number of subjects and track their gazes directly in a lengthy study.

Recent research presents an alternative. Machine learning has greatly improved the adherence of computational models of gaze fixation to real human gaze behavior. The current state-of-the-art is DeepGaze IIE @deepgazeiie, boasting \~80% adherence to real gaze distributions on benchmarks #footnote[The DeepGaze IIE paper refers to this adherence percentage as the ratio of a model's "information gain" to that of a gold standard. This computation is described further in the Background section.]. These models may now provide more signal of human behavior than noise, and so might provide useful feedback in creative processes. Crucially, they do so in a matter of seconds. We believe that these models are an important step towards greater automation and more compelling experiences in visual media.

Alas, these models may not be applicable to many styles of visual media. Gaze prediction models are notably biased towards candid photography: Matthias Kümmerer and Matthias Bethge @annurev-vision show that all leading gaze prediction models utilize transfer learning from other problem domains, primarily object recognition. Researchers do this becase of the lack of training data for the gaze recognition task, relative to the object recognition task.  The object recognition task prioritizes candid photography over stylistic representations due to the assumption that object recognition applications usually involve unaltered images captured with a camera, but this assumption does not hold true for the gaze prediction task. This assumption is implicitly encoded into most of the training data a gaze prediction model will see, and so the model's performance may suffer when generalizing to stylized images.

Furthermore, a study by Zhaohui Che, Ali Borji, Guangtao Zhai, Xiongkuo Min, Guodong Guo and Patrick Le Callet @gaze-transformations shows that image transformations influence gaze fixations, and effects of transformations on gaze behavior may be difficult to model. This fact is problematic, because modern visual productions use multiple steps of image processing and rendering for stylistic effect. Each step represents an image transformation, and a possible degradation of gaze prediction performance. Demonstrating robust performance, or the lack thereof, for a wide variety of transformations, not to mention combinations of transformations, is a daunting task if human trials are required; we hope to prove a more scalable approach.

We recognize three classifications for common image transformations, and we contribute simple heuristic functions for each class of transformation. These heuristics are meant to be statistically significant signals for how well a model's prediction adheres to a real gaze distribution. Notably, computing these heuristics does not require the collection of real gaze distributions, making them cheap to run. Our analysis of the heuristics utilizes real gaze distribution data on images before and after  transformation provided by Che et al.

While contributing these heuristics, we establish a loose mathematical definition for prediction adherence, which affords us the ability to determine whether a transformation is "composable". We argue that composable transformations can be applied to an image with lesser concern for gaze prediction degradation.

= RELATED WORK
The effects of image transformations--including cropping, rotation, skewing, and edge detection--on gaze behavior are studied by Che et al. @gaze-transformations. Their paper is motivated by the possibility of augmenting gaze prediction datasets with transformed images. We extend their work by repeating their experiments using current state-of-the-art models which had not been published at the time. We describe this experiment in detail in the Method section. We use the results of these experiments to study the effectiveness of computed heuristics on model predictions for transformed images.

The effects of digital image alterations on gaze behavior have also been studied by Rodrigo Quian Quiroga and Carlos Pedreira @quianquiroga, who performed an experiment collecting gaze fixations before and after manual Photoshop edits were made to photographs of paintings. Insufficient explanation for the intention behind edits makes it difficult to formally and generally describe the image transformations they studied. Instead, we study computationally-modeled image transformations, such that we can describe the effects they produce on gaze behavior more formally and generally, and so that we can study a greater scale of data which does not require human decisions for each image.

= BACKGROUND
The foundations we build upon are reviewed comprehensively by Kümmerer et al. @annurev-vision. We will briefly describe key terms and concepts.

In the process of studying an image, the human eye will occasionally jump to a new fixation point in what is called a "saccade". Both measured and predicted gaze can be represented by a two-dimensional probability distribution, analagous to a monotone image. The image is sometimes referred to as a "saliency map", but we will invent the term "gaze density" in order to disambiguate from the many different uses of the term "saliency map".

In gaze densities, pixels with the highest intensity are those most likely to be the next fixation target after a saccade for an arbitrary person under "free-viewing" conditions (which means the person has not been instructed to search for an element of the image).

In order the measure how gaze densities diverge, we use the Kullback-Leibler divergence, which measures the entropy between two probability distributions in terms of the number of bits required to describe the difference between those distributions.

We will also use another measure of divergence, called "information gain". Information gain is defined by Matthias Kümmerer, Thomas S. A. Wallis, and Matthias Bethge @information-gain to be the KL-divergence from a baseline prediction called the "center bias", which is produced by averaging all gaze densities for the dataset of interest. The center bias is the best prediction possible for any selected image from the dataset of interest, if one has no knowledge of the image's contents. It is called the center bias because viewers tend to look towards the center of an image.

We use information gain in our method because it describes the complexity of a gaze density, and when combined with KL-divergence, it can tell us whether the divergence between two gaze densities is due to a loss or gain in complexity, or neither.

The most widely adopted benchmark for gaze prediction model performance is the MIT/Tuebingen Saliency Benchmark @mit-tuebingen, which utilizes information gain, KL-divergence, and several other metrics to compare the performance of submitted models. The benchmark also compares models to a gold standard: they leave a random subject out of the real gaze data from the validation set, and produce gaze densities using the remaining data. These gaze densities diverge slightly from the validation data because of the omission of a subject, but they approximate a "best possible" prediction. The current top contender on the benchmark, DeepGaze IIE @deepgazeiie, achieves roughly 80% of the information gain of the gold standard, with the center bias being defined as 0% information gain. The model achieves roughly equivalent performance in other metrics.

We will perform experiments using the DeepGaze IIE model and the UNISAL model @unisal, which achieves rougly 70% of the information gain of the gold standard (less than DeepGaze IIE), but which has the advantages of running at or near real-time for video and being much smaller in memory. The models studied previously by Che et al. @gaze-transformations for image transformations achieved roughly 60% of information gain of the gold standard on the MIT/Tuebingen Saliency Benchmark.

The outputs of the particular gaze prediction models we will be experimenting with, as well as the gaze data collected by Che et al., are unnormalized gaze densities. In order to compare divergence between gaze densities, we may normalize to a probability distribution or a log probability distribution. For our experiment, we will be performing our calculations for both of these normalizations in parallel, because they each provide complementary information. Compared to a normal probability distribution, a log probability distribution lowers the influence of shared or differing area between two gaze densities when computing divergence, and raises the influence of shared or differing global maxima. See figure 1.

#figure(
  image("../figures/density_examples.png"),
  caption: "In this figure, we see a small section of the same gaze density, normalized to a probability distribution (left) and a log probability distribution (right). The intensity of pixels have been clipped in order to improve visibility: the bright spot on the right contains much higher intensity values than the brightest pixels on the left."
)

= METHOD
We wish to produce a method which shows whether a predicted gaze density adheres to the real gaze density when considering an image that is transformed with one of three classes of transformation.

First, let us define some terms. We will refer to the gaze density of an image before a transformation as the "prior density", and the gaze density of the image after a transformation as the "subsequent density". We will refer to KL-divergence as "divergence" for brevity. We define "adherence" to be a condition for a transformation, such that the divergence of a prior density to its appropriate real gaze density is greater than or equal (to within 5% of the divergence value) to the corresponding subsequent density to its real gaze density, for 95% of images tested under the transformation.

We call such a transformation a "composable" transformation. Assuming a transformation has been tested for adherence over a large number and large variety of images, we can be confident that the gaze prediction model understands the transformation and its effects on gaze behavior. Composable transformations may be applied freely to an image without concern for degrading the model's performance.

TODO:
- Obviously, if divergence is low, then it stands to reason that greater subsequent adherence is likely.
- If divergence is higher, but information asymmetry is low, then greater subsequent adherence might be the case: we should study.
- Check divergence, information asymmetry, and a  product of both metrics for binary classification and for error bound/trend
- Figure out where to set confidence bounds for adherence

We wish to provide evidence that two metrics, computed on the prior density and the subsequent density, are statistically significant heuristics for whether the adherence of the subsequent density is roughly equal or greater than the adherence of the prior density to its corresponding real gaze density.

We will argue that the condition of greater subsequent adherence implies that the transformation does not degrade the performance of a gaze prediction model. We will refer to this condition as "composability". Transformations that have been proven composable for a gaze prediction model can be applied to images freely, without the need for consideration of the effects on the model's performance.

The first metric is KL-divergence, which will referred to henceforth simply as "divergence". The second is the difference in information gain between the two distributions. We we will invent the term "information asymmetry" to refer to this difference in information gain.

We apply the stylization at relative strengths, such that we can study whether a stylization has non-linear effects on the prediction.

So that we can compare the effects of different stylizations, we normalize the KL-divergence and information gain difference by the size of the image and the least-squares difference between the stylization and the original image. The resulting metric tells us the effect size per pixel of the image, per pixel intensity value altered by the stylization. We compute the mean, median, and standard deviation of these normalized metrics across all images for a given stylization.

We use the divergence and information gain difference resulting from random noise as a control group for the comparison of effects of stylizations. If a stylization produces lower metrics than random noise, the stylization has little effect on the prediction produced by the model, and vice versa for higher metrics.

If a stylization produces metrics significantly different from random noise, whether lower or higher, and if an explanation for the effect based on human visual behavior can't be produced, it warrants further study and training for the model. The same can be said for metrics which are not significantly different, contrary to expectation from human visual behavior.

TODO: 
- add problem statement to the end of background
- add the paper references to MIT benchmark

#bibliography("paper.bib", title: "REFERENCES")