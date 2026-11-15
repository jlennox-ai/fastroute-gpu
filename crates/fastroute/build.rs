fn main() {
    // On Linux, `--gpu` loads NVRTC from the folder of the executable, as Windows does: users put
    // the libraries from NVIDIA's NVRTC archive next to `fastroute`. DT_RPATH, not DT_RUNPATH,
    // because NVRTC loads its builtins library itself and only DT_RPATH covers those loads.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,--disable-new-dtags,-rpath,$ORIGIN");
    }
}
