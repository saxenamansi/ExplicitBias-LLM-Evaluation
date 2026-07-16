


api_key = "EMPTY"  # vLLM ignores this but LLooM requires a value

l = wb.lloom(
    df=advice_df,
    id_col="narrative_idx",
    text_col="advice_text",
    distill_model=Model(setup_fn=setup_llm_fn, fn=call_llm_fn,
                         name="Qwen/Qwen2.5-14B-Instruct", cost=[0, 0],
                         rate_limit=(300, 10), context_window=32768, api_key=api_key),
    cluster_model=EmbedModel(setup_fn=setup_embed_fn, fn=call_embed_fn,
                              name="bge-large-en-v1.5", cost=0,
                              batch_size=2048, api_key=api_key),
    synth_model=Model(setup_fn=setup_llm_fn, fn=call_llm_fn,
                       name="Qwen/Qwen2.5-14B-Instruct", cost=[0, 0],
                       rate_limit=(20, 10), context_window=32768, api_key=api_key),
    score_model=Model(setup_fn=setup_llm_fn, fn=call_llm_fn,
                       name="Qwen/Qwen2.5-14B-Instruct", cost=[0, 0],
                       rate_limit=(300, 10), context_window=32768, api_key=api_key),
)