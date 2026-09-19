// sram_r_register.v
// Static Rotor Auxiliary Memory (SRAM-R) Register Block with Pinning Support

module sram_r_register #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 128 // Rotor format: {scalar[31:0], bivector_xy[31:0], bivector_yz[31:0], bivector_zx[31:0]}
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  we,
    input  wire                  pin_en,
    input  wire [ADDR_WIDTH-1:0] addr,
    input  wire [DATA_WIDTH-1:0] wdata,
    output reg  [DATA_WIDTH-1:0] rdata,
    output reg                   pinned_status
);

    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];
    reg [0:(1<<ADDR_WIDTH)-1] pinned_flags;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pinned_flags <= 0;
            rdata <= 0;
            pinned_status <= 0;
        end else begin
            if (we) begin
                mem[addr] <= wdata;
                pinned_flags[addr] <= pin_en;
            end
            rdata <= mem[addr];
            pinned_status <= pinned_flags[addr];
        end
    end

endmodule
