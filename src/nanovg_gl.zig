const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const use_webgl = builtin.cpu.arch.isWasm();
const gl = @import("zgl");

const logger = std.log.scoped(.nanovg_gl);

const nvg = @import("nanovg.zig");
const internal = @import("internal.zig");

pub const Options = struct {
    antialias: bool = false,
    stencil_strokes: bool = false,
    debug: bool = false,
};

pub fn init(allocator: Allocator, options: Options) !nvg {
    const gl_context = try GLContext.init(allocator, options);

    const params = internal.Params{
        .user_ptr = gl_context,
        .edge_antialias = options.antialias,
        .renderCreate = renderCreate,
        .renderCreateTexture = renderCreateTexture,
        .renderDeleteTexture = renderDeleteTexture,
        .renderUpdateTexture = renderUpdateTexture,
        .renderGetTextureSize = renderGetTextureSize,
        .renderViewport = renderViewport,
        .renderCancel = renderCancel,
        .renderFlush = renderFlush,
        .renderFill = renderFill,
        .renderStroke = renderStroke,
        .renderTriangles = renderTriangles,
        .renderDelete = renderDelete,
    };
    return nvg{
        .ctx = try internal.Context.init(allocator, params),
    };
}

const GLContext = struct {
    allocator: Allocator,
    options: Options,
    shader: Shader,
    view: [2]f32,
    textures: ArrayList(Texture),
    texture_id: i32 = 0,
    vert_buf: gl.Buffer = .invalid,
    calls: ArrayList(Call),
    paths: ArrayList(Path),
    verts: ArrayList(internal.Vertex),
    uniforms: ArrayList(FragUniforms),

    fn init(allocator: Allocator, options: Options) !*GLContext {
        var self = try allocator.create(GLContext);
        self.* = GLContext{
            .allocator = allocator,
            .options = options,
            .shader = undefined,
            .view = .{ 0, 0 },
            .textures = ArrayList(Texture).init(allocator),
            .calls = ArrayList(Call).init(allocator),
            .paths = ArrayList(Path).init(allocator),
            .verts = ArrayList(internal.Vertex).init(allocator),
            .uniforms = ArrayList(FragUniforms).init(allocator),
        };
        return self;
    }

    fn deinit(ctx: *GLContext) void {
        ctx.shader.delete();
        ctx.textures.deinit();
        ctx.calls.deinit();
        ctx.paths.deinit();
        ctx.verts.deinit();
        ctx.uniforms.deinit();
        ctx.allocator.destroy(ctx);
    }

    fn castPtr(ptr: *anyopaque) *GLContext {
        return @ptrCast(*GLContext, @alignCast(@alignOf(*GLContext), ptr));
    }

    fn allocTexture(ctx: *GLContext) !*Texture {
        var found_tex: ?*Texture = null;
        for (ctx.textures.items) |*tex| {
            if (tex.id == 0) {
                found_tex = tex;
                break;
            }
        }
        if (found_tex == null) {
            found_tex = try ctx.textures.addOne();
        }
        const tex = found_tex.?;
        tex.* = std.mem.zeroes(Texture);
        ctx.texture_id += 1;
        tex.id = ctx.texture_id;

        return tex;
    }

    fn findTexture(ctx: *GLContext, id: i32) ?*Texture {
        for (ctx.textures.items) |*tex| {
            if (tex.id == id) return tex;
        }
        return null;
    }
};

const ShaderType = enum(u2) {
    fill_gradient,
    fill_image,
    simple,
    image,
};

const Shader = struct {
    prog: gl.Program,
    frag: gl.Shader,
    vert: gl.Shader,

    view_loc: ?u32,
    tex_loc: ?u32,
    colormap_loc: ?u32,
    frag_loc: ?u32,

    fn create(shader: *Shader, header: [:0]const u8, vertsrc: [:0]const u8, fragsrc: [:0]const u8) !void {
        var status: gl.Int = undefined;

        shader.* = std.mem.zeroes(Shader);

        const prog = gl.createProgram();
        const vert = gl.createShader(.vertex);
        const frag = gl.createShader(.fragment);
        gl.shaderSource(vert, 2, &.{ header, vertsrc });
        gl.shaderSource(frag, 2, &.{ header, fragsrc });

        vert.compile();
        status = vert.get(.compile_status);
        if (status != 1) {
            printShaderErrorLog(vert, "shader", "vert");
            return error.ShaderCompilationFailed;
        }

        frag.compile();
        status = frag.get(.compile_status);
        if (status != 1) {
            printShaderErrorLog(frag, "shader", "frag");
            return error.ShaderCompilationFailed;
        }

        prog.attach(vert);
        prog.attach(frag);

        gl.bindAttribLocation(prog, 0, "vertex");
        gl.bindAttribLocation(prog, 1, "tcoord");

        prog.link();
        status = prog.get(.link_status);
        if (status != 1) {
            printProgramErrorLog(prog, "shader");
            return error.ProgramLinkingFailed;
        }

        shader.prog = prog;
        shader.vert = vert;
        shader.frag = frag;

        shader.getUniformLocations();
    }

    fn delete(shader: Shader) void {
        if (shader.prog != .invalid) gl.deleteProgram(shader.prog);
        if (shader.vert != .invalid) gl.deleteShader(shader.vert);
        if (shader.frag != .invalid) gl.deleteShader(shader.frag);
    }

    fn getUniformLocations(shader: *Shader) void {
        shader.view_loc = gl.getUniformLocation(shader.prog, "viewSize");
        shader.tex_loc = gl.getUniformLocation(shader.prog, "tex");
        shader.colormap_loc = gl.getUniformLocation(shader.prog, "colormap");
        shader.frag_loc = gl.getUniformLocation(shader.prog, "frag");
    }

    fn printShaderErrorLog(shader: gl.Shader, name: []const u8, shader_type: []const u8) void {
        var buf: [512]u8 = undefined;
        const log = gl.getShaderInfoLog(shader, &buf);
        logger.err("Shader {s}/{s} error:\n{s}", .{ name, shader_type, log });
    }

    fn printProgramErrorLog(program: gl.Program, name: []const u8) void {
        var buf: [512]u8 = undefined;
        const log = gl.getProgramInfoLog(program, &buf);
        logger.err("Program {s} error:\n{s}", .{ name, log });
    }
};

const Texture = struct {
    id: i32,
    tex: gl.Texture,
    width: i32,
    height: i32,
    tex_type: internal.TextureType,
    flags: nvg.ImageFlags,
};

const Blend = struct {
    src_rgb: gl.BlendFactor,
    dst_rgb: gl.BlendFactor,
    src_alpha: gl.BlendFactor,
    dst_alpha: gl.BlendFactor,

    fn fromOperation(op: nvg.CompositeOperationState) Blend {
        return .{
            .src_rgb = convertBlendFuncFactor(op.src_rgb),
            .dst_rgb = convertBlendFuncFactor(op.dst_rgb),
            .src_alpha = convertBlendFuncFactor(op.src_alpha),
            .dst_alpha = convertBlendFuncFactor(op.dst_alpha),
        };
    }

    fn convertBlendFuncFactor(factor: nvg.BlendFactor) gl.BlendFactor {
        return switch (factor) {
            .zero => gl.BlendFactor.zero,
            .one => gl.BlendFactor.one,
            .src_color => gl.BlendFactor.src_color,
            .one_minus_src_color => gl.BlendFactor.one_minus_src_color,
            .dst_color => gl.BlendFactor.dst_color,
            .one_minus_dst_color => gl.BlendFactor.one_minus_dst_color,
            .src_alpha => gl.BlendFactor.src_alpha,
            .one_minus_src_alpha => gl.BlendFactor.one_minus_src_alpha,
            .dst_alpha => gl.BlendFactor.dst_alpha,
            .one_minus_dst_alpha => gl.BlendFactor.one_minus_dst_alpha,
            .src_alpha_saturate => gl.BlendFactor.src_alpha_saturate,
        };
    }
};

const call_type = enum {
    none,
    fill,
    convexfill,
    stroke,
    triangles,
};

const Call = struct {
    call_type: call_type,
    image: i32,
    colormap: i32,
    path_offset: u32,
    path_count: u32,
    triangle_offset: u32,
    triangle_count: u32,
    uniform_offset: u32,
    blend_func: Blend,

    fn fill(call: Call, ctx: *GLContext) void {
        const paths = ctx.paths.items[call.path_offset..][0..call.path_count];

        // Draw shapes
        gl.enable(.stencil_test);
        defer gl.disable(.stencil_test);
        gl.stencilMask(0xff);
        gl.stencilFunc(.always, 0x0, 0xff);
        gl.colorMask(false, false, false, false);

        // set bindpoint for solid loc
        setUniforms(ctx, call.uniform_offset, 0, 0);

        gl.stencilOpSeparate(.front, .keep, .keep, .incr_wrap);
        gl.stencilOpSeparate(.back, .keep, .keep, .decr_wrap);
        gl.disable(.cull_face);
        for (paths) |path| {
            gl.drawArrays(.triangle_fan, path.fill_offset, path.fill_count);
        }
        gl.enable(.cull_face);

        // Draw anti-aliased pixels
        gl.colorMask(true, true, true, true);

        setUniforms(ctx, call.uniform_offset + 1, call.image, call.colormap);

        if (ctx.options.antialias) {
            gl.stencilFunc(.equal, 0x00, 0xff);
            gl.stencilOp(.keep, .keep, .keep);
            // Draw fringes
            for (paths) |path| {
                gl.drawArrays(.triangle_strip, path.stroke_offset, path.stroke_count);
            }
        }

        // Draw fill
        gl.stencilFunc(.not_equal, 0x0, 0xff);
        gl.stencilOp(.zero, .zero, .zero);
        gl.drawArrays(.triangle_strip, call.triangle_offset, call.triangle_count);
    }

    fn convexFill(call: Call, ctx: *GLContext) void {
        const paths = ctx.paths.items[call.path_offset..][0..call.path_count];

        setUniforms(ctx, call.uniform_offset, call.image, call.colormap);

        for (paths) |path| {
            gl.drawArrays(.triangle_fan, path.fill_offset, path.fill_count);
            // Draw fringes
            if (path.stroke_count > 0) {
                gl.drawArrays(.triangle_strip, path.stroke_offset, path.stroke_count);
            }
        }
    }

    fn stroke(call: Call, ctx: *GLContext) void {
        const paths = ctx.paths.items[call.path_offset..][0..call.path_count];

        if (ctx.options.stencil_strokes) {
            gl.enable(.stencil_test);
            defer gl.disable(.stencil_test);

            gl.stencilMask(0xff);

            // Fill the stroke base without overlap
            gl.stencilFunc(.equal, 0x0, 0xff);
            gl.stencilOp(.keep, .keep, .incr);
            setUniforms(ctx, call.uniform_offset + 1, call.image, call.colormap);
            for (paths) |path| {
                gl.drawArrays(.triangle_strip, path.stroke_offset, path.stroke_count);
            }

            // Draw anti-aliased pixels.
            setUniforms(ctx, call.uniform_offset, call.image, call.colormap);
            gl.stencilFunc(.equal, 0x00, 0xff);
            gl.stencilOp(.keep, .keep, .keep);
            for (paths) |path| {
                gl.drawArrays(.triangle_strip, path.stroke_offset, path.stroke_count);
            }

            // Clear stencil buffer.
            gl.colorMask(false, false, false, false);
            gl.stencilFunc(.always, 0x0, 0xff);
            gl.stencilOp(.zero, .zero, .zero);
            for (paths) |path| {
                gl.drawArrays(.triangle_strip, path.stroke_offset, path.stroke_count);
            }
            gl.colorMask(true, true, true, true);
        } else {
            setUniforms(ctx, call.uniform_offset, call.image, call.colormap);
            // Draw Strokes
            for (paths) |path| {
                gl.drawArrays(.triangle_strip, path.stroke_offset, path.stroke_count);
            }
        }
    }

    fn triangles(call: Call, ctx: *GLContext) void {
        setUniforms(ctx, call.uniform_offset, call.image, call.colormap);
        gl.drawArrays(.triangles, call.triangle_offset, call.triangle_count);
    }
};

const Path = struct {
    fill_offset: u32,
    fill_count: u32,
    stroke_offset: u32,
    stroke_count: u32,
};

fn maxVertCount(paths: []const internal.Path) usize {
    var count: usize = 0;
    for (paths) |path| {
        count += path.fill.len;
        count += path.stroke.len;
    }
    return count;
}

fn xformToMat3x4(m3: *[12]f32, t: *const [6]f32) void {
    m3[0] = t[0];
    m3[1] = t[1];
    m3[2] = 0;
    m3[3] = 0;
    m3[4] = t[2];
    m3[5] = t[3];
    m3[6] = 0;
    m3[7] = 0;
    m3[8] = t[4];
    m3[9] = t[5];
    m3[10] = 1;
    m3[11] = 0;
}

fn premulColor(c: nvg.Color) nvg.Color {
    return .{ .r = c.r * c.a, .g = c.g * c.a, .b = c.b * c.a, .a = c.a };
}

const FragUniforms = struct {
    scissor_mat: [12]f32, // matrices are actually 3 vec4s
    paint_mat: [12]f32,
    inner_color: nvg.Color,
    outer_color: nvg.Color,
    scissor_extent: [2]f32,
    scissor_scale: [2]f32,
    extent: [2]f32,
    radius: f32,
    feather: f32,
    stroke_mult: f32,
    stroke_thr: f32,
    tex_type: f32,
    shaderType: f32,

    fn fromPaint(frag: *FragUniforms, paint: *nvg.Paint, scissor: *internal.Scissor, width: f32, fringe: f32, stroke_thr: f32, ctx: *GLContext) i32 {
        var invxform: [6]f32 = undefined;

        frag.* = std.mem.zeroes(FragUniforms);

        frag.inner_color = premulColor(paint.inner_color);
        frag.outer_color = premulColor(paint.outer_color);

        if (scissor.extent[0] < -0.5 or scissor.extent[1] < -0.5) {
            std.mem.set(f32, &frag.scissor_mat, 0);
            frag.scissor_extent[0] = 1;
            frag.scissor_extent[1] = 1;
            frag.scissor_scale[0] = 1;
            frag.scissor_scale[1] = 1;
        } else {
            _ = nvg.transformInverse(&invxform, &scissor.xform);
            xformToMat3x4(&frag.scissor_mat, &invxform);
            frag.scissor_extent[0] = scissor.extent[0];
            frag.scissor_extent[1] = scissor.extent[1];
            frag.scissor_scale[0] = @sqrt(scissor.xform[0] * scissor.xform[0] + scissor.xform[2] * scissor.xform[2]) / fringe;
            frag.scissor_scale[1] = @sqrt(scissor.xform[1] * scissor.xform[1] + scissor.xform[3] * scissor.xform[3]) / fringe;
        }

        std.mem.copy(f32, &frag.extent, &paint.extent);
        frag.stroke_mult = (width * 0.5 + fringe * 0.5) / fringe;
        frag.stroke_thr = stroke_thr;

        if (paint.image.handle != 0) {
            const tex = ctx.findTexture(paint.image.handle) orelse return 0;
            if (tex.flags.flip_y) {
                var m1: [6]f32 = undefined;
                var m2: [6]f32 = undefined;
                nvg.transformTranslate(&m1, 0, frag.extent[1] * 0.5);
                nvg.transformMultiply(&m1, &paint.xform);
                nvg.transformScale(&m2, 1, -1);
                nvg.transformMultiply(&m2, &m1);
                nvg.transformTranslate(&m1, 0, -frag.extent[1] * 0.5);
                nvg.transformMultiply(&m1, &m2);
                _ = nvg.transformInverse(&invxform, &m1);
            } else {
                _ = nvg.transformInverse(&invxform, &paint.xform);
            }
            frag.shaderType = @intToFloat(f32, @enumToInt(ShaderType.fill_image));

            if (tex.tex_type == .rgba) {
                frag.tex_type = if (tex.flags.premultiplied) 0 else 1;
            } else if (paint.colormap.handle == 0) {
                frag.tex_type = 2;
            } else {
                frag.tex_type = 3;
            }
        } else {
            frag.shaderType = @intToFloat(f32, @enumToInt(ShaderType.fill_gradient));
            frag.radius = paint.radius;
            frag.feather = paint.feather;
            _ = nvg.transformInverse(&invxform, &paint.xform);
        }

        xformToMat3x4(&frag.paint_mat, &invxform);

        return 1;
    }
};

fn setUniforms(ctx: *GLContext, uniform_offset: u32, image: i32, colormap: i32) void {
    const frag = &ctx.uniforms.items[uniform_offset];
    gl.uniform4fv(ctx.shader.frag_loc, @ptrCast([*][4]f32, frag)[0..11]);

    if (colormap != 0) {
        if (ctx.findTexture(colormap)) |tex| {
            gl.activeTexture(.texture_1);
            tex.tex.bind(.@"2d");
            gl.activeTexture(.texture_0);
        }
    }

    if (image != 0) {
        if (ctx.findTexture(image)) |tex| {
            tex.tex.bind(.@"2d");
        }
    }
    // // If no image is set, use empty texture
    // if (tex == NULL) {
    // 	tex = glnvg__findTexture(gl->dummyTex);
    // }
    // glnvg__bindTexture(tex != NULL ? tex->tex : 0);
}

fn renderCreate(uptr: *anyopaque) !void {
    const ctx = GLContext.castPtr(uptr);

    const vertSrc = @embedFile("glsl/fill.vert");
    const fragSrc = @embedFile("glsl/fill.frag");
    const fragHeader = if (ctx.options.antialias) "#define EDGE_AA 1\n" else "";
    try ctx.shader.create(fragHeader, vertSrc, fragSrc);

    ctx.vert_buf = gl.genBuffer();

    // Some platforms does not allow to have samples to unset textures.
    // Create empty one which is bound when there's no texture specified.
    // ctx.dummyTex = glnvg__renderCreateTexture(NVG_TEXTURE_ALPHA, 1, 1, 0, NULL);
}

fn renderCreateTexture(uptr: *anyopaque, tex_type: internal.TextureType, w: i32, h: i32, flags: nvg.ImageFlags, data: ?[*]const u8) !i32 {
    const ctx = GLContext.castPtr(uptr);
    var tex: *Texture = try ctx.allocTexture();

    tex.tex = gl.genTexture();
    tex.width = w;
    tex.height = h;
    tex.tex_type = tex_type;
    tex.flags = flags;
    tex.tex.bind(.@"2d");

    switch (tex_type) {
        .none => {},
        .alpha => gl.textureImage2D(.@"2d", 0, .red, @intCast(usize, w), @intCast(usize, h), .red, .unsigned_byte, data),
        .rgba => gl.textureImage2D(.@"2d", 0, .rgba, @intCast(usize, w), @intCast(usize, h), .rgba, .unsigned_byte, data),
    }

    if (flags.generate_mipmaps) {
        if (flags.nearest) {
            gl.texParameter(.@"2d", .min_filter, .nearest);
        } else {
            gl.texParameter(.@"2d", .min_filter, .linear_mipmap_linear);
        }
    } else {
        if (flags.nearest) {
            gl.texParameter(.@"2d", .min_filter, .nearest);
        } else {
            gl.texParameter(.@"2d", .min_filter, .linear);
        }
    }

    if (flags.nearest) {
        gl.texParameter(.@"2d", .mag_filter, .nearest);
    } else {
        gl.texParameter(.@"2d", .mag_filter, .linear);
    }

    if (flags.repeat_x) {
        gl.texParameter(.@"2d", .wrap_s, .repeat);
    } else {
        gl.texParameter(.@"2d", .wrap_s, .clamp_to_edge);
    }

    if (flags.repeat_y) {
        gl.texParameter(.@"2d", .wrap_t, .repeat);
    } else {
        gl.texParameter(.@"2d", .wrap_t, .clamp_to_edge);
    }

    if (flags.generate_mipmaps) {
        gl.generateMipmap(.@"2d");
    }

    return tex.id;
}

fn renderDeleteTexture(uptr: *anyopaque, image: i32) void {
    const ctx = GLContext.castPtr(uptr);
    const tex = ctx.findTexture(image) orelse return;
    if (tex.tex != .invalid) tex.tex.delete();
    tex.* = std.mem.zeroes(Texture);
}

fn renderUpdateTexture(uptr: *anyopaque, image: i32, x_arg: i32, y: i32, w_arg: i32, h: i32, data_arg: ?[*]const u8) i32 {
    _ = x_arg;
    _ = w_arg;
    const ctx = GLContext.castPtr(uptr);
    const tex = ctx.findTexture(image) orelse return 0;

    // No support for all of skip, need to update a whole row at a time.
    const color_size: u32 = if (tex.tex_type == .rgba) 4 else 1;
    const y0: u32 = @intCast(u32, y * tex.width);
    const data = @ptrCast([*]const u8, &data_arg.?[y0 * color_size]);
    const x = 0;
    const w = @intCast(usize, tex.width);

    tex.tex.bind(.@"2d");
    switch (tex.tex_type) {
        .none => {},
        .alpha => gl.texSubImage2D(.@"2d", 0, x, @intCast(usize, y), w, @intCast(usize, h), .red, .unsigned_byte, data),
        .rgba => gl.texSubImage2D(.@"2d", 0, x, @intCast(usize, y), w, @intCast(usize, h), .rgba, .unsigned_byte, data),
    }
    gl.bindTexture(.invalid, .@"2d");

    return 1;
}

fn renderGetTextureSize(uptr: *anyopaque, image: i32, w: *i32, h: *i32) i32 {
    const ctx = GLContext.castPtr(uptr);
    const tex = ctx.findTexture(image) orelse return 0;
    w.* = tex.width;
    h.* = tex.height;
    return 1;
}

fn renderViewport(uptr: *anyopaque, width: f32, height: f32, devicePixelRatio: f32) void {
    const ctx = GLContext.castPtr(uptr);
    ctx.view[0] = width;
    ctx.view[1] = height;
    _ = devicePixelRatio;
}

fn renderCancel(uptr: *anyopaque) void {
    const ctx = GLContext.castPtr(uptr);
    ctx.verts.clearRetainingCapacity();
    ctx.paths.clearRetainingCapacity();
    ctx.calls.clearRetainingCapacity();
    ctx.uniforms.clearRetainingCapacity();
}

fn renderFlush(uptr: *anyopaque) void {
    const ctx = GLContext.castPtr(uptr);

    if (ctx.calls.items.len > 0) {
        // Setup required GL state.
        gl.useProgram(ctx.shader.prog);

        gl.enable(.cull_face);
        gl.cullFace(.back);
        gl.frontFace(.ccw);
        gl.enable(.blend);
        gl.disable(.depth_test);
        gl.disable(.scissor_test);
        gl.colorMask(true, true, true, true);
        gl.stencilMask(0xffffffff);
        gl.stencilOp(.keep, .keep, .keep);
        gl.stencilFunc(.always, 0, 0xffffffff);
        gl.activeTexture(.texture_0);
        gl.bindTexture(.invalid, .@"2d");

        ctx.vert_buf.bind(.array_buffer);
        gl.bufferData(.array_buffer, internal.Vertex, ctx.verts.items, .stream_draw);
        gl.enableVertexAttribArray(0);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(0, 2, .float, false, @sizeOf(internal.Vertex), 0);
        gl.vertexAttribPointer(1, 2, .float, false, @sizeOf(internal.Vertex), 2 * @sizeOf(f32));

        // Set view and texture just once per frame.
        gl.uniform1i(ctx.shader.tex_loc, 0);
        gl.uniform1i(ctx.shader.colormap_loc, 1);
        gl.uniform2fv(ctx.shader.view_loc, &.{ctx.view});

        for (ctx.calls.items) |call| {
            gl.blendFuncSeparate(call.blend_func.src_rgb, call.blend_func.dst_rgb, call.blend_func.src_alpha, call.blend_func.dst_alpha);
            switch (call.call_type) {
                .none => {},
                .fill => call.fill(ctx),
                .convexfill => call.convexFill(ctx),
                .stroke => call.stroke(ctx),
                .triangles => call.triangles(ctx),
            }
        }

        gl.disableVertexAttribArray(0);
        gl.disableVertexAttribArray(1);
        gl.disable(.cull_face);
        gl.bindBuffer(.invalid, .array_buffer);
        gl.useProgram(.invalid);
        gl.bindTexture(.invalid, .@"2d");
    }

    // Reset calls
    ctx.verts.clearRetainingCapacity();
    ctx.paths.clearRetainingCapacity();
    ctx.calls.clearRetainingCapacity();
    ctx.uniforms.clearRetainingCapacity();
}

fn renderFill(uptr: *anyopaque, paint: *nvg.Paint, composite_operation: nvg.CompositeOperationState, scissor: *internal.Scissor, fringe: f32, bounds: [4]f32, paths: []const internal.Path) void {
    const ctx = GLContext.castPtr(uptr);

    const call = ctx.calls.addOne() catch return;
    call.* = std.mem.zeroes(Call);

    call.call_type = .fill;
    call.triangle_count = 4;
    if (paths.len == 1 and paths[0].convex) {
        call.call_type = .convexfill;
        call.triangle_count = 0; // Bounding box fill quad not needed for convex fill
    }
    ctx.paths.ensureUnusedCapacity(paths.len) catch return;
    call.path_offset = @intCast(u32, ctx.paths.items.len);
    call.path_count = @intCast(u32, paths.len);
    call.image = paint.image.handle;
    call.colormap = paint.colormap.handle;
    call.blend_func = Blend.fromOperation(composite_operation);

    // Allocate vertices for all the paths.
    const maxverts = maxVertCount(paths) + call.triangle_count;
    ctx.verts.ensureUnusedCapacity(maxverts) catch return;

    for (paths) |path| {
        const copy = ctx.paths.addOneAssumeCapacity();
        copy.* = std.mem.zeroes(Path);
        if (path.fill.len > 0) {
            copy.fill_offset = @intCast(u32, ctx.verts.items.len);
            copy.fill_count = @intCast(u32, path.fill.len);
            ctx.verts.appendSliceAssumeCapacity(path.fill);
        }
        if (path.stroke.len > 0) {
            copy.stroke_offset = @intCast(u32, ctx.verts.items.len);
            copy.stroke_count = @intCast(u32, path.stroke.len);
            ctx.verts.appendSliceAssumeCapacity(path.stroke);
        }
    }

    // Setup uniforms for draw calls
    if (call.call_type == .fill) {
        // Quad
        call.triangle_offset = @intCast(u32, ctx.verts.items.len);
        ctx.verts.appendAssumeCapacity(.{ .x = bounds[2], .y = bounds[3], .u = 0.5, .v = 1.0 });
        ctx.verts.appendAssumeCapacity(.{ .x = bounds[2], .y = bounds[1], .u = 0.5, .v = 1.0 });
        ctx.verts.appendAssumeCapacity(.{ .x = bounds[0], .y = bounds[3], .u = 0.5, .v = 1.0 });
        ctx.verts.appendAssumeCapacity(.{ .x = bounds[0], .y = bounds[1], .u = 0.5, .v = 1.0 });

        call.uniform_offset = @intCast(u32, ctx.uniforms.items.len);
        ctx.uniforms.ensureUnusedCapacity(2) catch return;
        // Simple shader for stencil
        const frag = ctx.uniforms.addOneAssumeCapacity();
        frag.* = std.mem.zeroes(FragUniforms);
        frag.stroke_thr = -1.0;
        frag.shaderType = @intToFloat(f32, @enumToInt(ShaderType.simple));
        // Fill shader
        _ = ctx.uniforms.addOneAssumeCapacity().fromPaint(paint, scissor, fringe, fringe, -1.0, ctx);
    } else {
        call.uniform_offset = @intCast(u32, ctx.uniforms.items.len);
        ctx.uniforms.ensureUnusedCapacity(1) catch return;
        // Fill shader
        _ = ctx.uniforms.addOneAssumeCapacity().fromPaint(paint, scissor, fringe, fringe, -1.0, ctx);
    }
}

fn renderStroke(uptr: *anyopaque, paint: *nvg.Paint, composite_operation: nvg.CompositeOperationState, scissor: *internal.Scissor, fringe: f32, strokeWidth: f32, paths: []const internal.Path) void {
    const ctx = GLContext.castPtr(uptr);

    const call = ctx.calls.addOne() catch return;
    call.* = std.mem.zeroes(Call);

    call.call_type = .stroke;
    ctx.paths.ensureUnusedCapacity(paths.len) catch return;
    call.path_offset = @intCast(u32, ctx.paths.items.len);
    call.path_count = @intCast(u32, paths.len);
    call.image = paint.image.handle;
    call.colormap = paint.colormap.handle;
    call.blend_func = Blend.fromOperation(composite_operation);

    // Allocate vertices for all the paths.
    const maxverts = maxVertCount(paths);
    ctx.verts.ensureUnusedCapacity(maxverts) catch return;

    for (paths) |path| {
        const copy = ctx.paths.addOneAssumeCapacity();
        copy.* = std.mem.zeroes(Path);
        if (path.stroke.len > 0) {
            copy.stroke_offset = @intCast(u32, ctx.verts.items.len);
            copy.stroke_count = @intCast(u32, path.stroke.len);
            ctx.verts.appendSliceAssumeCapacity(path.stroke);
        }
    }

    if (ctx.options.stencil_strokes) {
        // Fill shader
        call.uniform_offset = @intCast(u32, ctx.uniforms.items.len);
        ctx.uniforms.ensureUnusedCapacity(2) catch return;
        _ = ctx.uniforms.addOneAssumeCapacity().fromPaint(paint, scissor, fringe, fringe, -1, ctx);
        _ = ctx.uniforms.addOneAssumeCapacity().fromPaint(paint, scissor, strokeWidth, fringe, 1.0 - 0.5 / 255.0, ctx);
    } else {
        // Fill shader
        call.uniform_offset = @intCast(u32, ctx.uniforms.items.len);
        _ = ctx.uniforms.ensureUnusedCapacity(1) catch return;
        _ = ctx.uniforms.addOneAssumeCapacity().fromPaint(paint, scissor, strokeWidth, fringe, -1, ctx);
    }
}

fn renderTriangles(uptr: *anyopaque, paint: *nvg.Paint, comp_op: nvg.CompositeOperationState, scissor: *internal.Scissor, fringe: f32, verts: []const internal.Vertex) void {
    const ctx = GLContext.castPtr(uptr);

    const call = ctx.calls.addOne() catch return;
    call.* = std.mem.zeroes(Call);

    call.call_type = .triangles;
    call.image = paint.image.handle;
    call.colormap = paint.colormap.handle;
    call.blend_func = Blend.fromOperation(comp_op);

    call.triangle_offset = @intCast(u32, ctx.verts.items.len);
    call.triangle_count = @intCast(u32, verts.len);
    ctx.verts.appendSlice(verts) catch return;

    call.uniform_offset = @intCast(u32, ctx.uniforms.items.len);
    const frag = ctx.uniforms.addOne() catch return;
    _ = frag.fromPaint(paint, scissor, 1, fringe, -1, ctx);
    frag.shaderType = @intToFloat(f32, @enumToInt(ShaderType.image));
}

fn renderDelete(uptr: *anyopaque) void {
    const ctx = GLContext.castPtr(uptr);
    ctx.deinit();
}
