# Key Logistics

**All deadlines 8:00 pm Eastern**

- Submit topic and group preferences by Jan 18
- Groups assigned by Jan 19
- **Proposal:** due Jan 25. (1%)
  - Expect feedback & TA assignment by Feb 1.
- **First Check-in:** Last week of February. (1%)
  - Progress expected.
  - By appointment with assigned TA - can be virtual.
- **Second Check-in:** Last week of March. Significant progress expected. (1%)
  - By appointment with assigned TA - can be virtual.
- **In-class presentations:**
  - Slides Due: Apr 8
  - Presentations: Apr 10, Apr 14, and Apr 17, during class hours (5%)
  - In-person attendance mandatory on all three presentation days. (2%)
- **Final Deiverables:** due Apr 17
  - Project summary, contributions tracking, peer review (5%)
  - GitHub repository (modular and documented) (5%)

# Overview

The aim of the final project is to allow you to pick one topic in AI to pursue in more depth
than course assignments permit and build something cool related to it from the ground up.
The project should be interesting (and hopefully useful to someone) - but you should also
consider feasibility when proposing a project. You might start looking for ideas in your daily
life - look for problems that you might want to solve using techniques covered in this course.
If you can turn your solution into something that lasts and is used actively beyond the final
project deadline, that would be a fantastic outcome!

I encourage you to be ambitious. There are no1 restrictions on the topic of the project (as
long as it is relevant to AI). That said, I strongly advise against projects that will rely on
LLMs/transformer architectures as a core feature of the pipeline - for several reasons: a) in
most projects of this nature, the LLM/transformer ends up doing most of the heavy lifting and
does not contribute meaningfully to the intended learning outcomes in this course, b) training
these models from scratch is infeasible with available compute infrastructure,
and fine-tuning existing models is not considered sufficient work for a group project, and c)
most projects involving LLMs/transformers tend to be highly contrived, repetitive or derivative,
and add little to no value to one’s résumé. I would much rather have students pick an ‘easier’
topic, but build the AI infrastructure from scratch in order to gain a solid understanding of
foundational methods.

The best projects are often those that arise organically - think not about what solution you
would like to implement, but simply start by identifying a problem that you would like to apply
AI towards. Tackling a problem that you personally face in your day-to-day life will lead to a
much more motivated project, and lead to much higher satisfaction. A common mistake I see students
make is looking for datasets on Kaggle, etc., and then designing a project idea that conveniently
fits this dataset. Such projects end up being limited in scope, and not faring well in evaluations.
Similarly, projects that are proposed by identifying an approach - say, a specific idea/technique a
student has prior experience with - and then fitting a problem around this technique tend to fail
at a significantly higher rate.

Think ahead of what we’ve already covered in class, and feel free to talk to me about
your ideas! Projects involving creative use of LLMs (not as the central aspect of the
project, but perhaps as a data generator, etc.), deep learning, and reinforcement learning
are particularly fun - and pose unique challenges that make them a worthwhile learning experience.
The next few sections detail what is expected from you at each stage of the project throughout the
semester, and include some examples of excellent projects from past versions of this course.

## Project Proposal

The first step is a short (1-page) write-up of your project idea. Your proposal should con-
tain a brief overview of what you’re planning to do, and why. Identify a problem that you
think can reasonably be tackled within a semester, and motivate it sufficiently. Class time
on Jan 23 (Friday) will be dedicated to project-proposal ideation with the TA team.
Identify and summarize existing approaches that have tried solving the same problem,
with references/links as appropriate. It is absolutely okay to not know which AI approach
would be the best solution at this point in the semester (especially since many of you are
new to the field). We will be happy to make recommendations for how to go about solving
the problem of your choice. You should then start to read ahead of lecture content and
build an understanding of the suggested technique(s) as a group. The proposal counts
for 1% out of the 20% project grade. Here are some cool past project ideas:

- Playing the NYT Connections game
- Combine existing instrumental music into novel tracks
- Translating ASL using webcam input
- Assist veterans in dealing with VA paperwork
- Optimizing city layouts to minimize travel time
- Equitable dynamic loan pricing

## Project Tracking

To keep track of project progress and individual contributions to the project, each team will
maintain an up-to-date GitHub repository connected to a GitHub Project. GitHub Projects
is a Kanban-style project tracking solution with seamless integration with repositories, and
allows teams to set up project road maps, assign tasks to members, track completion, etc.
This will help us both during check-ins (more below) as well as grading. The project session
in class on Jan 23 will also cover project tracking, and a tutorial will be shared with
the class on Campuswire.

Please note that each member’s final project grade will individually be based on the extent
of their contributions as tracked by commits, the Kanban board, and peer-evaluations, and
can be as low as 0 for non-participating members.

## Check-ins

Each group will be assigned a TA, whom they are required to meet with twice during the
semester, by appointment, on the dates specified on page 1. All team members must
be present for these check-ins. The check-in will be a 10-15 minute chat about how
your project is coming along, and will help us identify any shortcomings or hurdles that
need to be dealt with before the final deliverables are due. Please come prepared with a
summary of your research/approach thus far, an updated GitHub repository and project-
tracker showing individual contributions to the project, and any questions you may have.

This check-in will be scheduled outside lecture hours, and by appointment. Ideally, all
group members should be present for this check-in. This also gives us a chance to ensure
that all members in the group contributed equally to the project (more on this in later
sections). These check-ins count for 2% out of the 20% project grade.

## Final Presentation

Each group will present their work through an in-class presentation, lasting no longer
than 6 minutes, with an additional 2-3 minutes for questions. Presentations must include
a demo of the project; this gives you a chance to impress the audience and show off your
work. In the interest of minimizing turnaround time, all demos must be pre-recorded and
embedded in your presentation. To do so, upload your demo videos privately to YouTube
(or another similar service) and embed the web-hosted video into your presentation -
this will minimize both the submission file size for your PPT, as well as the probability of
things going wrong during the presentation. All group members must be present and are
expected to participate equally in the presentation. Points will be assigned on the basis
of clarity of presentation (both visual and content-related), formatting of results, ability
to answer relevant questions, and adherence to the time limit, for a total of 5% of your
course grade. A more detailed rubric will be shared with you closer to the presentations.
Attendance will be recorded, and will make up 2% of your overall course grade.

## Final Deliverables

### Project Summary

The primary deliverable at the end of the semester will be an executive summary of the
project. This will be a 1-2 page document that covers the motivation for the problem,
summarizes the implemented approach briefly, and highlights key results and challenges.
The document also contains a link to the GitHub repository and summarizes individual
contributions. A template for the project summary will be shared with the class.

### GitHub Repo

The implementation of your course project should be submitted as a public GitHub reposi-
tory (or with the instructor and assigned TA added as contributors). The repository should
be well-organized with modular, re-usable, and properly documented code. The reposi-
tory should contain a detailed ReadMe, which provides instructions for end-users to set
up an equivalent Python environment, execute your code to reproduce your results, and
also explains the organization of your code.

Throughout the semester, each student must commit their work to this repository to es-
tablish a history of contributions, which will be a factor in your final grade. Please ensure
that you contribute novel source code for the AI methods of your project, and not just
items such as visualization, environment setup, and hyperparameter-tuning.

Jupyter notebooks are not an acceptable submission format for the final project, but may
be used solely for any data cleaning and preprocessing steps. If so, include the notebook
and detailed documentation for the user in your GitHub repository.

### Group Member Ratings, Contributions

Throughout the semester, students may raise concerns about underperforming team
members through Canvas (“Project Team Concern Report” in the Navigation sidebar).
These reports may be filed as information only, or request direct intervention from me or
the TA team. At the end of the semester, each student will receive an overall project grade
proportional to the extent of their contributions (this can be a zero). Each student will also
have an opportunity to submit peer ratings for all of their group members. This rating will
be submitted individually and will only be visible to the instructor and TAs. Consistent neg-
ative ratings from the rest of your group will have consequences on your project grade.
All students are also expected to contribute directly to the actual implementation in terms
of code, which will be evaluated through git history.

Working in teams can be challenging; however, it is a vital skill to gain at University. Be
it academia or industry, most work with any real-world impact is the result of a collabora-
tion. Most challenging situations in group projects are easily avoided through open and
direct communication. To track your team’s progress, and to keep each other accountable,
please make sure to use GitHub Projects regularly. Having an objective source of contri-
bution history can be an effective tool in conflict resolution, should the need arise. Please
also use the TA team’s expertise and your instructor’s availability to your advantage, and
feel free to reach out if we can help with anything!

# Grading

Grades for the project will be assigned based on the novelty, creativity, and applicability of
the solution, the execution of the methods proposed, the ability to explain key findings and
relate them to the problem motivation, the extent to which shortcomings are addressed,
and individual contributions. In addition, the repository will be graded for adherence to
expectations detailed above. Combined, the repository and the executive summary con-
tribute 10% to your overall grade.
