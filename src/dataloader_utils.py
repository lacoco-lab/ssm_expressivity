import copy
from src.dataloader import (
    TomitaCorpus,
    DyckCorpus,
    ParityCorpus,
    NonStarFreeCorpus,
    ShuffleCorpus,
    CounterCorpus,
    BooleanExprCorpus,
    StarFreeCorpus,
    StarFreePostLanguageCorpus
)


def create_corpus_tomita(params):
    # Retrieve all the relevant parameters
    debug, is_leak, num_params = params['debug'], params['leak'], params['num_par']
    train_size, test_size, num_val_bins = params['training_size'], params['test_size'], params['num_val_bins']
    lower_window, upper_window, len_incr = params['lower_window'], params['upper_window'], params['len_incr']

    if not is_leak:
        corpus = TomitaCorpus(num_params, lower_window, upper_window, train_size + test_size, unique=True, debug=debug)
        # Train and validation created together, hence separate them out first
        train_corpus = copy.deepcopy(corpus)
        train_corpus.source, train_corpus.target = corpus.source[:train_size], corpus.target[:train_size]

        val_corpus = copy.deepcopy(corpus)
        val_corpus.source, val_corpus.target = corpus.source[train_size:], corpus.target[train_size:]
        val_corpus_bins = [val_corpus]

        # Prepare to make validation sets for greater window sizes
        lower_window = upper_window + 1
        upper_window = upper_window + len_incr

        for i in range(num_val_bins - 1):
            print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
            val_corpus_bin = TomitaCorpus(num_params, lower_window, upper_window, test_size, unique=True, debug=debug)
            val_corpus_bins.append(val_corpus_bin)
            lower_window = upper_window
            upper_window = upper_window + params['len_incr']

    else:
        train_corpus = TomitaCorpus(num_params, lower_window, upper_window, train_size,
                                    unique=False, leak=True, debug=params['debug'])
        val_corpus_bins = [TomitaCorpus(num_params, lower_window, upper_window, test_size, unique=True, leak=True, debug=debug)]

        lower_window = upper_window + 1
        upper_window = upper_window + len_incr

        for i in range(num_val_bins-1):
            val_corpus_bin = TomitaCorpus(num_params, lower_window, upper_window, test_size, unique=True, leak=True, debug=debug)
            val_corpus_bins.append(val_corpus_bin)
    
    return train_corpus, val_corpus_bins


def create_corpus_parity(params):
    # Retrieve all the relevant parameters
    train_size, test_size, num_val_bins = params['training_size'], params['test_size'], params['num_val_bins']
    lower_window, upper_window, len_incr = params['lower_window'], params['upper_window'], params['len_incr']
    # Create the corpus in go for a validation and training set
    corpus = ParityCorpus(lower_window, upper_window, train_size + test_size)

    # separate out the training set
    train_corpus = copy.deepcopy(corpus)
    train_corpus.source, train_corpus.target = corpus.source[:train_size], corpus.target[:train_size]

    # Separate out the validation set
    val_corpus = copy.deepcopy(corpus)
    val_corpus.source, val_corpus.target = corpus.source[train_size:], corpus.target[train_size:]
    val_corpus_bins = [val_corpus]

    # Prepare for making other valiation sets
    lower_window = upper_window + 1
    upper_window = upper_window + len_incr

    # Create one less validation bin, since validation bin already made above    
    for i in range(num_val_bins - 1):
        print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
        val_corpus_bin = ParityCorpus(lower_window, upper_window, test_size)
        val_corpus_bins.append(val_corpus_bin)
        lower_window = upper_window
        upper_window = upper_window + len_incr

    return train_corpus, val_corpus_bins

    
def create_corpus_non_star_free(params):
    ### ABABStar, AAStar, AnStarA2 ###
    # Retrieve the relevant parameters
    curr_lang, num_params = params['lang_class'], params['num_par']
    train_size, test_size, num_val_bins = params['training_size'], params['test_size'], params['num_val_bins']
    lower_window, upper_window, len_incr = params['lower_window'], params['upper_window'], params['len_incr']

    # Create the training and 1st validation bin
    train_corpus = NonStarFreeCorpus(curr_lang, num_params, lower_window, upper_window, train_size)
    val_corpus_bins = [NonStarFreeCorpus(curr_lang, num_params, lower_window, upper_window, test_size, unique = True)]

    # Increment for subsequent bins
    lower_window = upper_window + 1
    upper_window = upper_window + len_incr

    # Create one less validation bin, since validation bin already made above    
    for i in range(num_val_bins - 1):
        val_corpus_bin = NonStarFreeCorpus(curr_lang, num_params, lower_window, upper_window, test_size, unique = True)
        val_corpus_bins.append(val_corpus_bin)

        lower_window = upper_window
        upper_window += len_incr
    return train_corpus, val_corpus_bins


def create_corpus_dyck(params):
    # Retrieve all the relevant parameters
    lower_depth, upper_depth = params['min_depth'], params['max_depth']
    p_val, q_val, num_params = params['p_val'], params['q_val'], params['num_par']
    bin1_lower_depth, bin1_upper_depth = params['bin1_lower_depth'], params['bin1_upper_depth']
    train_size, test_size, num_val_bins = params['training_size'], params['test_size'], params['num_val_bins']
    lower_window, upper_window, len_incr = params['lower_window'], params['upper_window'], params['len_incr']

    # Create the train corpus, and the first validation bin with almost identical parameters
    train_corpus = DyckCorpus(p_val, q_val, num_params, lower_window, upper_window, train_size, lower_depth, upper_depth)
    val_corpus_bins = [DyckCorpus(p_val, q_val, num_params, lower_window, upper_window, test_size, lower_depth, upper_depth)]

    # Increment the window sizes further for subsequent validation bins
    lower_window = upper_window + 2
    upper_window = upper_window + len_incr
    lower_depth = bin1_lower_depth
    upper_depth = bin1_upper_depth

    for i in range(num_val_bins-1):
        print("Generating Data for depths [{}, {}] and Lengths [{}, {}]".format(lower_depth, upper_depth, lower_window,
                                                                                upper_window))
        val_corpus_bin = DyckCorpus(p_val, q_val, num_params, lower_window, upper_window, test_size, lower_depth,
                                       upper_depth)
        val_corpus_bins.append(val_corpus_bin)

        # Increment the window sizes further for subsequent validation bins
        lower_window = upper_window + 2
        upper_window = upper_window + len_incr

    return train_corpus, val_corpus_bins

def create_corpus_shuffle(params):
    # Retrieve all the relevant parameters
    lower_depth, upper_depth = params['lower_depth'], params['upper_depth']
    p_val, q_val, num_params = params['p_val'], params['q_val'], params['num_par']
    bin1_lower_depth, bin1_upper_depth = params['bin_lower_depth'], params['bin_upper_depth']
    train_size, test_size, num_val_bins = params['training_size'], params['test_size'], params['num_val_bins']
    lower_window, upper_window, len_incr = params['lower_window'], params['upper_window'], params['len_incr']

    # Create the train corpus, and the first validation bin with almost identical parameters
    train_corpus = ShuffleCorpus(p_val, q_val, num_params, lower_window, upper_window, train_size, lower_depth, upper_depth)
    val_corpus_bins = [ShuffleCorpus(p_val, q_val, num_params, lower_window, upper_window, test_size, lower_depth, upper_depth)]
    
    # Vary the window sizes, and depth sizes for the other validation bins
    lower_window = upper_window + 2
    upper_window = upper_window + len_incr
    lower_depth = bin1_lower_depth
    upper_depth = bin1_upper_depth

    # Create one less validation bin, since validation bin already made above
    for i in range(num_val_bins - 1):
        print("Generating Data for depths [{}, {}] and Lengths [{}, {}]".format(lower_depth, upper_depth, lower_window, upper_window))
        val_corpus_bin = ShuffleCorpus(p_val, q_val, num_params, lower_window, upper_window, test_size, lower_depth, upper_depth)
        val_corpus_bins.append(val_corpus_bin)
        
        # Increment the window sizes further for subsequent validation bins
        lower_window = upper_window
        upper_window = upper_window + len_incr
    return train_corpus, val_corpus_bins


def create_corpus_counter(params):
    # Retrieve relevant parameters
    train_size, test_size = params['training_size'], params['test_size']
    num_params, num_val_bins = params['num_par'], params['num_val_bins']
    lower_window, upper_window, len_incr = params['lower_window'], params['upper_window'], params['len_incr']
    
    # Create training and validation corpuses
    train_corpus = CounterCorpus(num_params, lower_window, upper_window, train_size)
    val_corpus_bins = [CounterCorpus(num_params, lower_window, upper_window, test_size)]

    # Increment the window sizes for subsequent validation bins
    lower_window = upper_window + 1
    upper_window = upper_window + len_incr

    # Create one less validation bin, since validation bin already made above    
    for i in range(num_val_bins - 1):
        val_corpus_bin = CounterCorpus(num_params, lower_window, upper_window, test_size)
        val_corpus_bins.append(val_corpus_bin)
        
        lower_window = upper_window
        upper_window += len_incr
    return train_corpus, val_corpus_bins
    

def create_corpus_boolean(params):
    # Retrieve all the relevant parameters
    p_val, num_params = params['p_val'], params['num_par']
    train_size, test_size, num_val_bins = params['training_size'], params['test_size'], params['num_val_bins']
    lower_window, upper_window, len_incr = params['lower_window'], params['upper_window'], params['len_incr']

    # Create the corpus in one go for a validation and training set
    corpus = BooleanExprCorpus(p_val, num_params, lower_window, upper_window, train_size + test_size)
    # separate out the combined corpus into training and validation sets
    train_corpus = copy.deepcopy(corpus)
    train_corpus.source, train_corpus.target = corpus.source[:train_size], corpus.target[:train_size]

    val_corpus = copy.deepcopy(corpus)
    val_corpus.source, val_corpus.target = corpus.source[train_size:], corpus.target[train_size:]
    val_corpus_bins = [val_corpus]
    
    # Prepare lower and upper window for other validation bins
    lower_window = upper_window + 1
    upper_window = upper_window + len_incr

    # Create one less validation bin, since validation bin already made above
    for i in range(num_val_bins - 1):
        print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
        val_corpus_bin = BooleanExprCorpus(p_val, num_params, lower_window, upper_window, test_size)
        val_corpus_bins.append(val_corpus_bin)
        
        lower_window = upper_window
        upper_window = upper_window + len_incr
    return train_corpus, val_corpus_bins


def create_corpus_star_free(params):
    # Retrieve the relevant parameters
    curr_lang = params['lang_class']
    train_size, test_size, num_val_bins = params['training_size'], params['test_size'], params['num_val_bins']
    lower_window, upper_window, len_incr = params['lower_window'], params['upper_window'], params['len_incr']
    
    if curr_lang != 'StarFreeSpecial':
        num_params, keep_unique = params['num_par'], params['unique']
        # Create the corpus in one go for a validation and training set
        corpus = StarFreeCorpus(curr_lang, num_params, lower_window, upper_window, train_size + test_size, keep_unique)
        # separate the combined corpus into train and validation sets
        train_corpus = copy.deepcopy(corpus)
        train_corpus.source, train_corpus.target = corpus.source[:train_size], corpus.target[:train_size]
        
        val_corpus = copy.deepcopy(corpus)
        val_corpus.source, val_corpus.target = corpus.source[train_size:], corpus.target[train_size:]
        val_corpus_bins = [val_corpus]

        # Prepare upper window for other validation bins [Lower window unchanged unlike other cases]
        upper_window = upper_window + len_incr
        
        # Create one less validation bin, since validation bin already made above
        for i in range(num_val_bins - 1):
            print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
            val_corpus_bin = StarFreeCorpus(curr_lang, num_params, lower_window, upper_window, test_size, keep_unique)
            val_corpus_bins.append(val_corpus_bin)
            upper_window = upper_window + len_incr
    else: 
        mandatory, pre_choices, post_choices = params['mandatory'], params['pre_choices'], params['post_choices']
        # Create the corpus in one go for a validation and training set
        corpus = StarFreePostLanguageCorpus(mandatory, pre_choices, post_choices, lower_window, upper_window, train_size + test_size)
        # separate the combined corpus into train and validation sets
        train_corpus = copy.deepcopy(corpus)
        train_corpus.source, train_corpus.target = corpus.source[:train_size], corpus.target[:train_size]
        
        val_corpus = copy.deepcopy(corpus)
        val_corpus.source, val_corpus.target = corpus.source[train_size:], corpus.target[train_size:]
        val_corpus_bins = [val_corpus]

        # Prepare upper window for other validation bins [Lower window unchanged unlike other cases]
        upper_window = upper_window + len_incr
        
        # Create one less validation bin, since validation bin already made above
        for i in range(num_val_bins - 1):
            print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
            val_corpus_bin = StarFreePostLanguageCorpus(mandatory, pre_choices, post_choices, lower_window, upper_window, test_size)
            val_corpus_bins.append(val_corpus_bin)
            upper_window = upper_window + len_incr

    return train_corpus, val_corpus_bins