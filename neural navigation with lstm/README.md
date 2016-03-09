<h1>Neural mapping of navigational instructions to action sequences</h1>
<p> Based on the work of Hongyuan Mei, Mohit Bansal, and Matthew R. Walter in the paper <b>"Listen, Attend, and Walk: Neural Mapping
of Navigational Instructions to Action Sequences"</b> (link <a href="http://arxiv.org/abs/1506.04089">here</a>).</p>
<p>The implementation of the entire model made by the authors can be found in Hongyuan Mei's reposiroty <a href="https://github.com/hongyuan88/NeuralWalker">here</a> (coming soon).

<p>The paper proposes a neural encoder-decoder model with attention components for the task of natural language instruction following. The model draws "attention" from low and high-level representations of the instructions -a multi-level aligner-, improving accuracy of inferred directions.</p>
<p>The dataset used was the raw version of the SAIL route instruction dataset collected by MacMahon, Stankiewicz, and Kuipers (2006). The dataset contains 706 non-trivial navigational instruction paragraphs, produced by six instructors for 126 unique start and end position pairs spread evenly across three virtual worlds. Chen and Mooney (2011) segmented the data into individual sentences and paired each one of them with an action sequence. Check the original dataset <a href="http://www.cs.utexas.edu/users/ml/clamp/navigation/">here</a>.
</p>

References:
Mei, H.; Bansal, M.; and Walter, M.R.. 2016. Listen, Attend, and Walk: Neural Mapping
of Navigational Instructions to Action Sequences. In Proceedings of the Conference on Artificial Intelligence (AAAI) 2016.

MacMahon, M.; Stankiewicz, B.; and Kuipers, B. 2006. Walk the talk: Connecting language, knowledge, and action in route instructions. In Proceedings of the National Conference on Artificial Intelligence (AAAI) 2006.

Chen, D. L., and Mooney, R. J. 2011. Learning to interpret natural language navigation instructions from observations. In Proceedings of the National Conference on Artificial Intelligence (AAAI).