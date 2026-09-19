// pim_nvm_dma_pipeline.v
// PIM/NVM Asynchronous DMA Imprint Pipeline

module pim_nvm_dma_pipeline #(
    parameter DATA_WIDTH = 288 // 3x3 Metric Tensor x 32-bit float
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  dma_req,
    input  wire [31:0]           chart_addr,
    input  wire [DATA_WIDTH-1:0] metric_tensor_data,
    output reg                   dma_ack,
    output reg                   nvm_write_en,
    output reg  [31:0]           nvm_addr,
    output reg  [DATA_WIDTH-1:0] nvm_wdata
);

    reg [2:0] state;
    localparam IDLE  = 3'b000;
    localparam SYNC  = 3'b001;
    localparam WRITE = 3'b010;
    localparam DONE  = 3'b011;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            dma_ack <= 1'b0;
            nvm_write_en <= 1'b0;
            nvm_addr <= 32'd0;
            nvm_wdata <= 0;
        end else begin
            case (state)
                IDLE: begin
                    dma_ack <= 1'b0;
                    nvm_write_en <= 1'b0;
                    if (dma_req) begin
                        state <= SYNC;
                    end
                end
                SYNC: begin
                    nvm_addr <= chart_addr;
                    nvm_wdata <= metric_tensor_data;
                    state <= WRITE;
                end
                WRITE: begin
                    nvm_write_en <= 1'b1;
                    state <= DONE;
                end
                DONE: begin
                    nvm_write_en <= 1'b0;
                    dma_ack <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

endmodule
