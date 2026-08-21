import { app } from "../../scripts/app.js";

// "Always keep exactly one more slot than what's connected" pattern,
// scoped to mask_-prefixed inputs only. ColorcraftDebug also has a fixed
// required "colorcraft_debug" input in this.inputs alongside the dynamic
// ones -- sweeping that up in the same count/add/remove logic would
// misbehave, since it's never meant to be duplicated or removed.
app.registerExtension({
    name: "colorcraft.Debug",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ColorcraftDebug") return;

        nodeType.prototype.onConnectionsChange = function () {
            const maskInputs = this.inputs.filter((i) => i.name.startsWith("mask_"));
            const connected = maskInputs.filter((i) => i.link != null);
            const target = connected.length + 1;

            while (maskInputs.length < target) {
                this.addInput(`mask_${maskInputs.length}`, "COLORCRAFT_MASK");
                maskInputs.push(this.inputs[this.inputs.length - 1]);
            }
            while (maskInputs.length > target) {
                const last = maskInputs[maskInputs.length - 1];
                if (last.link != null) break;
                this.removeInput(this.inputs.indexOf(last));
                maskInputs.pop();
            }
        };
    },
});