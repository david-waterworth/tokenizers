use tokenizers::models::bpe::BPE;
use tokenizers::{DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper};
use tokenizers::{Model, TokenizerBuilder};

// #[test]
// fn bpe_values_after_training() {
//     let mut tokenizer = TokenizerBuilder::<
//         BPE,
//         NormalizerWrapper,
//         PreTokenizerWrapper,
//         PostProcessorWrapper,
//         DecoderWrapper,
//     >::default()
//     .with_model(
//         BPE::builder()
//             .unk_token("[UNK]".to_string())
//             .dropout(0.1)
//             .build()
//             .unwrap(),
//     )
//     .build()
//     .unwrap();
//     let mut trainer = tokenizer.get_model().get_trainer();
//     tokenizer
//         .train_from_files(&mut trainer, vec!["./data/small.txt".to_string()])
//         .unwrap();
//     assert_eq!(tokenizer.get_model().dropout, Some(0.1));
//     assert_eq!(tokenizer.get_model().unk_token, Some("[UNK]".to_string()));
// }

use tokenizers::models::unigram::{UnigramTrainer, Unigram};
use tokenizers::normalizers::{strip::Strip};
use tokenizers::pre_tokenizers::delimiter::CharDelimiterSplit;
#[test]
fn unigram_training() {
    let vocab_size: u32 = 100;
    let mut trainer = UnigramTrainer::builder()
        .show_progress(true)
        .vocab_size(vocab_size)
        .build()
        .unwrap();
    
    let model = Unigram::default();

    let mut tokenizer = TokenizerBuilder::<
        Unigram,
        Strip,
        CharDelimiterSplit,
        PostProcessorWrapper,
        DecoderWrapper,
    >::default()
    .with_model(model)
    .with_pre_tokenizer(Some(CharDelimiterSplit::new('-')))
    .with_normalizer(Some(Strip::new(true,true)))
    .build().unwrap();

    tokenizer
        .train_from_files(&mut trainer, vec!["/home/david/dev/tokenizer/tokenizers/tokenizers/tests/data/vocab.txt".to_string()]).unwrap()
        .save("/home/david/dev/tokenizer/tokenizers/tokenizers/tests/data/tokenizer.json", true).unwrap();
}