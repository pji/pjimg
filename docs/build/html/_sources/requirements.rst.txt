##################
pjimg Requirements
##################
The purpose of this document is to detail the requirements for
:mod:'pjimg', a Python image and animation generator. This is an
initial take for the purposes of planning. There may be additional
requirements or non-required features added in the future.


Purpose
=======
The purposes of :mod:`pjimg` are:

*   Generate images and animations that involve simple patterns interacting,
*   Be a more accurate name than :mod:`pjinoise` was,
*   Consolidate the separate "img*" modules.


Functional Requirements
=======================
The following are the functional requirements for :mod:`pjimg`:

*   Generate images and animations of a given size.


Technical Requirements
======================
The following are the technical requirements for :mod:`pjimg`:

*   Provide a common namespace for the "img*" modules.
*   Consolidate common utilities used across the "img*" modules.


Design Discussion
=================
The following is a deeper discussion of certain aspects of the
:mod:`pjimg` design. This primarily exists as a place to talk
through design challenges in order do find solutions. It is not
intended to be comprehensive nor even completely accurate to the
final design.


Why?
----
Good question. I split up :mod:`pjinoise`. Why am I bringing it back
with a different name?

The main reason is that the individual "img*" modules aren't very
useful on their own. I suppose they could be used in someone else's
image generation code, but is that really going to happen? Who
needs :mod:`imgwriter` who isn't implementing it on their own? It
just a wrapper around open-cv for file I/O.

Also, I just feel bad about using up the namespace of five different
modules if I ever put this on PYPI. I'm not sure I'll ever want to
do that. But, if I do, I'd rather it all be one name.


How Do I Make Images with It?
-----------------------------
:mod:`pjinoise` had a configuration syntax that could be used to
generate images. The "img*" scraped that in favor of making every
image a Python script. So, what now?

I'm probably going to stick with Python scripts. The configuration
syntax was too much of a pain to manage. That said :mod:`pjimg` can
still have some features that make writing those scripts easier.
And, who knows. Maybe a GUI can be set up for it eventually.


So, What Is :mod:`pjimg`?
-------------------------
It is:

*   A namespace for the "img*" modules.
*   A common module to handle code shared between those modules.
*   Possibly some workflow and UI tools to make working with
    the modules easier.
