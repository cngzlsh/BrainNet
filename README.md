<h1 class="code-line" data-line-start=0 data-line-end=1 ><a id="BrainNet_0"></a>BrainNet</h1>
<p class="has-line-data" data-line-start="1" data-line-end="4">This GitHub repository contains coding files for my MSc Project, entitled<br>
Approximation of Arbitrary Neuronal Network with Deep Learning<br>
Submitted as partial requirements of the MSc in Machine Learning in University College London.</p>
<p class="has-line-data" data-line-start="5" data-line-end="6">All files are in the <code>masters</code> branch:</p>
<ul>
<li class="has-line-data" data-line-start="6" data-line-end="7"><code>approx_bnn.py</code> contains 4 classes of approximate biological neural networks (ABNNs);</li>
<li class="has-line-data" data-line-start="7" data-line-end="8"><code>gen_data.py</code> contains code for generating synthetic spike trains and input patterns to feed into ABNNs;</li>
<li class="has-line-data" data-line-start="8" data-line-end="9"><code>models.py</code> contains DNN classes: MLP and RNN;</li>
<li class="has-line-data" data-line-start="9" data-line-end="10"><code>train.py</code> contains main training loop;</li>
<li class="has-line-data" data-line-start="10" data-line-end="11"><code>bvc.py</code> contains classes for the environment, roaming agent, the boundary vector cell and place cell network;</li>
<li class="has-line-data" data-line-start="11" data-line-end="13"><code>utils.py</code> contains a range of auxiliary functions, such as plotting;<br>
Folders:</li>
<li class="has-line-data" data-line-start="13" data-line-end="14"><code>approx_bnn_params</code> stores parameters for ABNNs, including individual transfer functions for each neuron;</li>
<li class="has-line-data" data-line-start="14" data-line-end="15"><code>dnn_params</code> stores parameters for trained DNNs;</li>
<li class="has-line-data" data-line-start="15" data-line-end="16"><code>data</code> stores input-output pairs generated from synthetic spike trains, ABNN and BVC network;</li>
<li class="has-line-data" data-line-start="16" data-line-end="17"><code>temp</code> stores temporary files and training results for plotting;</li>
<li class="has-line-data" data-line-start="17" data-line-end="19"><code>figures</code> stores saved figures<br>
Notebooks:</li>
<li class="has-line-data" data-line-start="19" data-line-end="20"><code>abnn_pattern_analysis</code> contains experiments on analysing synthetic neuronal inputs and outputs;</li>
<li class="has-line-data" data-line-start="20" data-line-end="21"><code>abnn_experiments</code> contains all experiments in the first part (ABNN);</li>
<li class="has-line-data" data-line-start="21" data-line-end="26"><code>bvc_experiments</code> contains all experiments in the second part (BVC model).<br>
To recreate the results, run<br>
<code>git clone https://github.com/cngzlsh/BrainNet.git --branch master</code><br>
And use the seed (1234) embedded at the beginning of each file.<br>
The codes are tested on a machine with the following specifications:</li>
<li class="has-line-data" data-line-start="26" data-line-end="27">Intel Core i7-11370H CPU (4-core, 8-threads), 40 GiB RAM, Nvidia RTX 3070 Laptop GPU (5,120-core, 8 GiB DRAM), 512 GiB SSD on Pop!_OS 22.04 LTS (Gnome version 42.4).</li>
</ul>
