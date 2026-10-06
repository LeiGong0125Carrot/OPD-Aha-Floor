"""PBD branch continuation loop: crop-view prompt + a fixed assistant prefix, continued for at most
`pbd_max_cont_len` tokens. Registered as `pbd_branch` (scripts/pbd_agent_loop.yaml)."""
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("pbd_branch")
class PBDBranchAgentLoop(SingleTurnAgentLoop):
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        prefix_ids = [int(t) for t in kwargs["pbd_prefix_ids"]]
        max_cont = int(kwargs["pbd_max_cont_len"])
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        prompt_ids = await self.apply_chat_template(messages, tools=self.tool_schemas, images=images, videos=videos)
        prompt_ids = list(prompt_ids) + prefix_ids          # generation prompt + student prefix (continued, no new turn)
        sp = dict(sampling_params); sp["max_tokens"] = max_cont
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sp, image_data=images, video_data=videos,
            )
        response_length = self._get_response_length()
        n = min(len(output.token_ids), response_length)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[:n],
            response_mask=[1] * n,
            response_logprobs=output.log_probs[:n] if output.log_probs else None,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )
