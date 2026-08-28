import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

BLUE, RED, GREY, LGREY = "#1f4e79", "#c0392b", "#5d6d7e", "#d5dbdb"
GREEN, ORANGE = "#1e8449", "#d68910"

def grid(ax, x0, y0, n, m, cell=0.3, fc="white", vals=None, fs=5.5,
         highlight=None, hfc="#aed6f1", ec=GREY):
    for i in range(n):
        for j in range(m):
            f = hfc if (highlight and (i, j) in highlight) else fc
            ax.add_patch(Rectangle((x0+j*cell, y0-(i+1)*cell), cell, cell,
                                   fc=f, ec=ec, lw=0.6))
            if vals is not None:
                ax.text(x0+(j+.5)*cell, y0-(i+.5)*cell, str(vals[i][j]),
                        ha="center", va="center", fontsize=fs)

def arw(ax, x1, y1, x2, y2, c=GREY, lw=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw))

def off(ax):
    ax.set_aspect("equal"); ax.axis("off")


# ============================================== FIG 1: pipeline (compact)
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 1.75))
    st = [("Input","128x128x3",1.05,BLUE),("Conv 32","128x128x32",1.05,GREEN),
          ("Pool","64x64x32",.82,GREY),("Conv 64","64x64x64",.82,GREEN),
          ("Pool","32x32x64",.62,GREY),("Conv 128","32x32x128",.62,GREEN),
          ("Pool","16x16x128",.46,GREY),("Conv 128","16x16x128",.46,GREEN),
          ("Pool","8x8x128",.34,GREY),("Dense","8192->128",.34,ORANGE),
          ("Softmax","4",.24,RED)]
    x = 0
    for i,(n,sh,h,c) in enumerate(st):
        w = .56
        ax.add_patch(Rectangle((x,-h/2), w, h, fc=c, ec="none", alpha=.8))
        ax.text(x+w/2, 0, n, ha="center", va="center", fontsize=5.3,
                color="white", weight="bold")
        ax.text(x+w/2, -h/2-.10, sh, ha="center", va="top", fontsize=5.0,
                color=BLUE)
        if i < len(st)-1: arw(ax, x+w+.02, 0, x+w+.26, 0)
        x += w+.25
    ax.set_xlim(-.15, x); ax.set_ylim(-.85, .70); off(ax)
    plt.savefig("c1_pipeline.pdf", bbox_inches="tight"); plt.close()


# ============================================== FIG 2: convolution 2x2
def fig_conv():
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.15))

    # (a) one convolution step
    ax = axes[0]
    inp=[[3,1,0,2,1],[0,2,1,3,0],[1,0,4,1,2],[2,1,0,2,3],[0,3,1,0,1]]
    ker=[[1,0,-1],[1,0,-1],[1,0,-1]]
    out=[[".",".","."],[".","-3","."],[".",".","."]]
    hl={(i,j) for i in range(1,4) for j in range(1,4)}
    grid(ax,0,1.55,5,5,vals=inp,highlight=hl)
    grid(ax,2.05,1.30,3,3,vals=ker,fc="#fadbd8")
    grid(ax,3.65,1.40,3,3,vals=out,highlight={(1,1)},hfc="#a9dfbf")
    ax.text(.75,1.68,"input 5x5",ha="center",fontsize=6,color=BLUE)
    ax.text(2.50,1.43,"kernel",ha="center",fontsize=6,color=RED)
    ax.text(4.10,1.53,"output 3x3",ha="center",fontsize=6,color=GREEN)
    arw(ax,1.60,1.00,2.00,1.00); arw(ax,3.00,1.00,3.60,1.00)
    ax.set_title("(a) one convolution step",fontsize=8)
    ax.set_xlim(-.15,4.8); ax.set_ylim(.25,1.85); off(ax)

    # (b) no padding
    ax = axes[1]
    grid(ax,0,1.55,5,5,highlight={(i,j) for i in range(3) for j in range(3)})
    grid(ax,2.05,1.40,3,3,fc="#a9dfbf")
    ax.text(.75,1.68,"5x5",ha="center",fontsize=6)
    ax.text(2.50,1.53,"3x3",ha="center",fontsize=6)
    ax.text(1.6,-.02,"output shrinks; border pixels\nseen by fewer positions",
            ha="center",va="top",fontsize=6,color=RED)
    ax.set_title("(b) P=0, s=1",fontsize=8)
    ax.set_xlim(-.15,3.3); ax.set_ylim(-.65,1.85); off(ax)

    # (c) same padding
    ax = axes[2]
    c=.3
    for i in range(7):
        for j in range(7):
            e=(i in (0,6)) or (j in (0,6))
            ax.add_patch(Rectangle((j*c,1.55-(i+1)*c),c,c,
                fc="#f2f3f4" if e else "white", ec=LGREY if e else GREY,lw=.6))
    grid(ax,2.45,1.40,5,5,fc="#a9dfbf")
    ax.text(1.05,1.68,"5x5 + zero border",ha="center",fontsize=6)
    ax.text(3.20,1.53,"5x5",ha="center",fontsize=6)
    ax.text(2.0,-.45,"spatial size preserved",ha="center",va="top",
            fontsize=6,color=GREEN)
    ax.set_title("(c) P=1, s=1  ('same')",fontsize=8)
    ax.set_xlim(-.15,4.1); ax.set_ylim(-.85,1.85); off(ax)

    # (d) stride 2
    ax = axes[3]
    grid(ax,0,1.55,5,5,highlight={(i,j) for i in range(3) for j in range(3)})
    for j in (0,2):
        ax.add_patch(Rectangle((j*.3,1.55-3*.3),3*.3,3*.3,fc="none",
                     ec=RED,lw=1.1,ls="--"))
    grid(ax,2.05,1.47,2,2,fc="#a9dfbf")
    ax.text(.75,1.68,"5x5",ha="center",fontsize=6)
    ax.text(2.35,1.60,"2x2",ha="center",fontsize=6)
    ax.text(1.5,-.02,"window jumps 2 px;\nresolution halved",
            ha="center",va="top",fontsize=6,color=GREY)
    ax.set_title("(d) P=0, s=2",fontsize=8)
    ax.set_xlim(-.15,3.1); ax.set_ylim(-.65,1.85); off(ax)

    plt.tight_layout()
    plt.savefig("c2_conv.pdf", bbox_inches="tight"); plt.close()


# ============================================== FIG 3: pooling + head
def fig_pool_head():
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.15),
                             gridspec_kw={"width_ratios":[1,1.7]})
    # (a) max pooling
    ax = axes[0]
    vals=[[1,3,2,4],[5,6,1,0],[2,1,7,3],[0,4,2,8]]
    cols=["#aed6f1","#f9e79f","#a9dfbf","#f5b7b1"]; c=.34
    for qi,(r0,c0) in enumerate([(0,0),(0,2),(2,0),(2,2)]):
        for i in range(2):
            for j in range(2):
                ax.add_patch(Rectangle(((c0+j)*c,1.5-(r0+i+1)*c),c,c,
                             fc=cols[qi],ec=GREY,lw=.6))
                ax.text((c0+j+.5)*c,1.5-(r0+i+.5)*c,vals[r0+i][c0+j],
                        ha="center",va="center",fontsize=6)
    outv=[[6,4],[4,8]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((2.1+j*c,1.16-i*c),c,c,fc=cols[i*2+j],
                         ec=GREY,lw=.8))
            ax.text(2.1+(j+.5)*c,1.16+c/2-i*c,outv[i][j],ha="center",
                    va="center",fontsize=6.5,weight="bold")
    arw(ax,1.45,1.15,2.05,1.15); ax.text(1.75,1.24,"max",ha="center",
        fontsize=6,color=RED)
    ax.text(1.4,-.05,"each 2x2 window keeps its maximum:\n"
            "75% discarded, no parameters",ha="center",va="top",fontsize=6,
            color=GREY)
    ax.set_title("(a) max pooling 2x2, s=2",fontsize=8)
    ax.set_xlim(-.15,3.1); ax.set_ylim(-.65,1.65); off(ax)

    # (b) classification head
    ax = axes[1]
    for k in range(4):
        o=k*.06
        ax.add_patch(Rectangle((.05+o,.72+o),.42,.42,fc="#a9dfbf",ec=GREY,
                     lw=.6,alpha=.9))
    ax.text(.38,.60,"8x8x128",ha="center",va="top",fontsize=6)
    for i in range(9):
        ax.add_patch(Rectangle((1.05,.52+i*.085),.14,.085,fc="#d6eaf8",
                     ec=GREY,lw=.5))
    ax.text(1.12,.44,"flatten\n8192",ha="center",va="top",fontsize=6)
    arw(ax,.72,.95,1.00,.95)
    for i in range(5):
        ax.add_patch(Rectangle((1.75,.68+i*.12),.14,.12,fc="#fdebd0",
                     ec=ORANGE,lw=.5))
    ax.text(1.82,.60,"dense\n128",ha="center",va="top",fontsize=6)
    arw(ax,1.28,.95,1.70,.95)
    probs=[.06,.81,.09,.04]; names=["Glioma","Meningioma","Pituitary","No tumour"]
    for i,(p,n) in enumerate(zip(probs,names)):
        y=1.24-i*.21
        ax.add_patch(Rectangle((3.05,y),p*.95,.14,fc=RED if p>.5 else "#f5b7b1",
                     ec="none"))
        ax.text(3.00,y+.07,n,ha="right",va="center",fontsize=5.8)
        ax.text(3.08+p*.95,y+.07,f"{p:.2f}",ha="left",va="center",fontsize=5.5,
                color=GREY)
    arw(ax,1.98,.95,2.35,.95)
    ax.text(2.16,1.04,"softmax",ha="center",fontsize=6,color=RED)
    ax.text(2.0,.10,"8192 x 128 + 128 = 1,048,704 parameters  =  81% of the "
            "network",ha="center",va="top",fontsize=6.2,color=RED)
    ax.set_title("(b) flatten, dense head, softmax",fontsize=8)
    ax.set_xlim(-.1,4.3); ax.set_ylim(-.15,1.55)
    ax.set_aspect("auto"); ax.axis("off")

    plt.tight_layout()
    plt.savefig("c3_pool_head.pdf", bbox_inches="tight"); plt.close()


# ============================================== FIG 4: why CNNs win (1x3)
def fig_why():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 2.2))
    shared=[RED,GREEN,BLUE]

    # (a) parameter sharing
    ax=axes[0]
    inx=np.linspace(0,3.0,7); outx=np.linspace(.5,2.5,4)
    for a in inx:
        for b in outx:
            ax.plot([a,b],[1.9,1.35],color=LGREY,lw=.35,zorder=1)
    ax.scatter(inx,[1.9]*7,s=26,color="#aed6f1",ec=GREY,lw=.5,zorder=3)
    ax.scatter(outx,[1.35]*4,s=26,color="#f9e79f",ec=GREY,lw=.5,zorder=3)
    ax.text(1.5,2.06,"dense: 28 unique weights",ha="center",fontsize=6.5,
            color=RED)
    for k,b in enumerate(outx):
        ctr=.5+k*.5
        idx=[i for i,a in enumerate(inx) if abs(a-ctr)<.55][:3]
        for c,i in enumerate(idx):
            ax.plot([inx[i],b],[.75,.25],color=shared[c%3],lw=.8,zorder=2,
                    alpha=.85)
    ax.scatter(inx,[.75]*7,s=26,color="#aed6f1",ec=GREY,lw=.5,zorder=3)
    ax.scatter(outx,[.25]*4,s=26,color="#a9dfbf",ec=GREY,lw=.5,zorder=3)
    ax.text(1.5,.95,"conv: 3 shared weights",ha="center",fontsize=6.5,
            color=GREEN)
    ax.text(1.5,-.12,"same colour = same weight",ha="center",va="top",
            fontsize=6,color=GREY,style="italic")
    ax.set_title("(a) parameter sharing",fontsize=8)
    ax.set_xlim(-.3,3.3); ax.set_ylim(-.45,2.25); ax.axis("off")

    # (b) receptive field
    ax=axes[1]
    c=.19
    rows=[(1.75,9,"input","#aed6f1"),(1.25,7,"conv 1: 3x3","#a9dfbf"),
          (.75,5,"conv 2: 5x5","#f9e79f"),(.25,3,"conv 3: 7x7","#f5b7b1")]
    for y,n,lab,col in rows:
        x0=-(n*c)/2
        for j in range(n):
            ax.add_patch(Rectangle((x0+j*c,y),c,c,fc=col,ec=GREY,lw=.5))
        ax.text(x0-.10,y+c/2,lab,ha="right",va="center",fontsize=6)
    for (y1,n1,_,_),(y2,n2,_,_) in zip(rows[:-1],rows[1:]):
        for e in (-1,1):
            ax.plot([e*(n1*c)/2,e*(n2*c)/2],[y1,y2+c],color=GREY,lw=.5,ls=":")
    ax.text(0,-.10,"depth composes receptive fields",ha="center",va="top",
            fontsize=6,color=GREY,style="italic")
    ax.set_title("(b) local receptive fields",fontsize=8)
    ax.set_xlim(-1.7,1.0); ax.set_ylim(-.45,2.15); ax.axis("off")

    # (c) equivariance
    ax=axes[2]
    def block(shift,y0):
        img=np.zeros((8,8)); img[2:5,1+shift:3+shift]=1
        feat=np.zeros((8,8)); feat[3,1+shift]=1
        return np.hstack([img,np.ones((8,1))*.15,feat])
    top=block(0,0); bot=block(3,0)
    ax.imshow(np.vstack([top,np.ones((1,17))*.05,bot]),cmap="Blues",
              vmin=0,vmax=1.3)
    ax.text(3.5,17.9,"input",ha="center",va="top",fontsize=6,color=BLUE)
    ax.text(13,17.9,"feature map",ha="center",va="top",fontsize=6,color=GREEN)
    ax.text(-.9,3.5,"original",ha="right",va="center",fontsize=6)
    ax.text(-.9,12.5,"shifted",ha="right",va="center",fontsize=6)
    ax.text(8,19.6,"response moves with the input:\n"
            "f(shift(x)) = shift(f(x))",ha="center",va="top",fontsize=6,
            color=GREY,style="italic")
    ax.set_title("(c) translation equivariance",fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    plt.tight_layout()
    plt.savefig("c4_why.pdf", bbox_inches="tight"); plt.close()


for f in [fig_pipeline, fig_conv, fig_pool_head, fig_why]:
    f(); print("done", f.__name__)
