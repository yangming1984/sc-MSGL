function [C,T]=hungarian(A)
[m,n]=size(A);
if (m~=n)
    error('HUNGARIAN: Cost matrix must be square!');
end
orig=A;
A=hminired(A);
[A,C,U]=hminiass(A);
while (U(n+1))
    LR=zeros(1,n);
    LC=zeros(1,n);
    CH=zeros(1,n);
    RH=[zeros(1,n) -1];
    SLC=[];
    r=U(n+1);
    LR(r)=-1;
    SLR=r;
    while (1)
        if (A(r,n+1)~=0)
            l=-A(r,n+1);
            if (A(r,l)~=0 & RH(r)==0)
                RH(r)=RH(n+1);
                RH(n+1)=r;
                CH(r)=-A(r,l);
            end
        else
            if (RH(n+1)<=0)
                [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR);
            end
            r=RH(n+1);
            l=CH(r);
            CH(r)=-A(r,l);
            if (A(r,l)==0)
                RH(n+1)=RH(r);
                RH(r)=0;
            end
        end
        while (LC(l)~=0)
            if (RH(r)==0)
                if (RH(n+1)<=0)
                    [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR);
                end
                r=RH(n+1);
            end
            l=CH(r);
            CH(r)=-A(r,l);
            if(A(r,l)==0)
                RH(n+1)=RH(r);
                RH(r)=0;
            end
        end
        if (C(l)==0)
            [A,C,U]=hmflip(A,C,LC,LR,U,l,r);
            break;
        else
            LC(l)=r;
            SLC=[SLC l];
            r=C(l);
            LR(r)=l;
            SLR=[SLR r];
        end
    end
end
T=sum(orig(logical(sparse(C,1:size(orig,2),1))));
function A=hminired(A)
[m,n]=size(A);
colMin=min(A);
A=A-colMin(ones(n,1),:);
rowMin=min(A')';
A=A-rowMin(:,ones(1,n));
[i,j]=find(A==0);
A(1,n+1)=0;
for k=1:n
    cols=j(k==i)';
    A(k,[n+1 cols])=[-cols 0];
end
function [A,C,U]=hminiass(A)
[n,np1]=size(A);
C=zeros(1,n);
U=zeros(1,n+1);
LZ=zeros(1,n);
NZ=zeros(1,n);
for i=1:n
	lj=n+1;
	j=-A(i,lj);
	while (C(j)~=0)
		lj=j;
		j=-A(i,lj);
		if (j==0)
			break;
		end
	end
	if (j~=0)
		C(j)=i;
		A(i,lj)=A(i,j);
		NZ(i)=-A(i,j);
		LZ(i)=lj;
		A(i,j)=0;
	else
		lj=n+1;
		j=-A(i,lj);
		while (j~=0)
			r=C(j);
			lm=LZ(r);
			m=NZ(r);
			while (m~=0)
				if (C(m)==0)
					break;
				end
				lm=m;
				m=-A(r,lm);
			end
			if (m==0)
				lj=j;
				j=-A(i,lj);
			else
				A(r,lm)=-j;
				A(r,j)=A(r,m);
				NZ(r)=-A(r,m);
				LZ(r)=j;
				A(r,m)=0;
				C(m)=r;
				A(i,lj)=A(i,j);
				NZ(i)=-A(i,j);
				LZ(i)=lj;
				A(i,j)=0;
				C(j)=i;
				break;
			end
		end
	end
end
r=zeros(1,n);
rows=C(C~=0);
r(rows)=rows;
empty=find(r==0);
U=zeros(1,n+1);
U([n+1 empty])=[empty 0];
function [A,C,U]=hmflip(A,C,LC,LR,U,l,r)
n=size(A,1);
while (1)
    C(l)=r;
    m=find(A(r,:)==-l);
    A(r,m)=A(r,l);
    A(r,l)=0;
    if (LR(r)<0)
        ...remove row from unassigned row list and return.
        U(n+1)=U(r);
        U(r)=0;
        return;
    else
        l=LR(r);
        A(r,l)=A(r,n+1);
        A(r,n+1)=-l;
        r=LC(l);
    end
end
function [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
n=size(A,1);
coveredRows=LR==0;
coveredCols=LC~=0;
r=find(~coveredRows);
c=find(~coveredCols);
m=min(min(A(r,c)));
A(r,c)=A(r,c)-m;
for j=c
    for i=SLR
        if (A(i,j)==0)
            if (RH(i)==0)
                RH(i)=RH(n+1);
                RH(n+1)=i;
                CH(i)=j;
            end
            row=A(i,:);
            colsInList=-row(row<0);
            if (length(colsInList)==0)
                l=n+1;
            else
                l=colsInList(row(colsInList)==0);
            end
            A(i,l)=-j;
        end
    end
end
r=find(coveredRows);
c=find(coveredCols);
[i,j]=find(A(r,c)<=0);
i=r(i);
j=c(j);
for k=1:length(i)
    lj=find(A(i(k),:)==-j(k));
    A(i(k),lj)=A(i(k),j(k));
    A(i(k),j(k))=0;
end
A(r,c)=A(r,c)+m;
